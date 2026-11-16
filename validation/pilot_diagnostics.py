"""
pilot_diagnostics.py – 100-sequence pilot diagnostics
=======================================================
Generates 100 sequences per split and reports:
  1. Track lifetime distribution (frames survived)
  2. Simultaneous target counts per frame
  3. Feature distributions (detection features & context)
  4. Crossing trajectory frequency (CA/CT mode counts)

Usage:
    python -m validation.pilot_diagnostics [--out-dir /tmp/pilot_diag]
"""

import sys
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import h5py
import tempfile
import time

from simulator.generate_dataset import generate_split, TRAIN_CONFIGS, OOD_CONFIGS, CONFIG_KEY_LIST
from config.sim_config import N_FRAMES_PER_SEQ, N_TARGET_MAX

N_PILOT = 100   # sequences per split
BASE_SEED = 777

SPLITS = {
    'train': {'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 0},
    'val':   {'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 1},
    'test':  {'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 2},
    'ood':   {'cfg_pool': OOD_CONFIGS,   'seed_offset': 3},
}


def generate_pilot(out_dir):
    """Generate HDF5 pilot files, return paths dict."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split, cfg in SPLITS.items():
        path = os.path.join(out_dir, f'{split}_pilot.h5')
        if os.path.exists(path):
            print(f'  [reuse] {split}_pilot.h5')
        else:
            t0 = time.time()
            generate_split(
                split_name  = split,
                n_sequences = N_PILOT,
                cfg_pool    = cfg['cfg_pool'],
                out_path    = path,
                base_seed   = BASE_SEED * 10 + cfg['seed_offset'],
                n_workers   = 1,
            )
            print(f'  [gen]   {split}_pilot.h5  ({time.time()-t0:.1f}s)')
        paths[split] = path
    return paths


def track_lifetime_stats(h5path):
    """
    Compute track lifetimes from state_ids / state_valid arrays.
    A track starts when its ID first appears and ends when it last appears.
    Returns array of lifetimes in frames.
    """
    with h5py.File(h5path, 'r') as f:
        ids   = f['state_ids'][:]      # (N, T, N_MAX)
        valid = f['state_valid'][:]    # (N, T, N_MAX)

    lifetimes = []
    N, T, _ = ids.shape
    for s in range(N):
        # all unique positive IDs in this sequence
        seq_ids = np.unique(ids[s][valid[s]])
        seq_ids = seq_ids[seq_ids > 0]
        for tid in seq_ids:
            # frames where this ID is active
            frames_active = np.any((ids[s] == tid) & valid[s], axis=-1)  # (T,)
            lifetime = int(frames_active.sum())
            lifetimes.append(lifetime)

    return np.array(lifetimes, dtype=int)


def simultaneous_target_stats(h5path):
    """Count simultaneous live targets per frame."""
    with h5py.File(h5path, 'r') as f:
        sv = f['state_valid'][:]   # (N, T, N_MAX)
    counts = sv.sum(axis=-1).ravel()  # (N*T,)
    return counts


def feature_stats(h5path):
    """
    Compute mean/std/min/max for each of the 8 detection features and 5 context features.
    Only valid (non-padded) detections.
    """
    FEAT_NAMES = ['R_m', 'vr_mps', 'az_deg', 'el_deg',
                  'beam_az', 'beam_el', 'dt_s', 'Q_dB']
    CTX_NAMES  = ['B_PRS_MHz', 'N_pil', 'T_obs_ms', 'P_tx_dBm', 'P_SI_dBm']

    with h5py.File(h5path, 'r') as f:
        dets   = f['detections'][:]   # (N, T, M, 8)
        valid  = f['det_valid'][:]    # (N, T, M)
        ctx    = f['contexts'][:]     # (N, T, 5)

    # detection features — only valid slots
    det_valid_flat = dets[valid]  # (K, 8)
    det_stats = {}
    for i, name in enumerate(FEAT_NAMES):
        v = det_valid_flat[:, i]
        det_stats[name] = dict(mean=float(v.mean()), std=float(v.std()),
                               min=float(v.min()), max=float(v.max()),
                               p5=float(np.percentile(v, 5)),
                               p95=float(np.percentile(v, 95)))

    # context features — all frames
    ctx_flat = ctx.reshape(-1, 5)
    ctx_stats = {}
    for i, name in enumerate(CTX_NAMES):
        v = ctx_flat[:, i]
        ctx_stats[name] = dict(mean=float(v.mean()), std=float(v.std()),
                               min=float(v.min()), max=float(v.max()))

    return det_stats, ctx_stats


def cfg_transition_rate(h5path):
    """Measure resource-config self-transition rate from cfg_keys."""
    with h5py.File(h5path, 'r') as f:
        ck = f['cfg_keys'][:]  # (N, T)
    n_self, n_total = 0, 0
    for s in range(ck.shape[0]):
        for t in range(1, ck.shape[1]):
            n_total += 1
            if ck[s, t] == ck[s, t-1]:
                n_self += 1
    return n_self / n_total if n_total else 0.0


def print_report(paths):
    print("\n" + "="*70)
    print("  RCATT-SIM PILOT DIAGNOSTICS  (N=100 sequences/split)")
    print("="*70)

    for split, h5path in paths.items():
        print(f"\n--- Split: {split.upper()} ---")

        # 1. Track lifetime
        lt = track_lifetime_stats(h5path)
        if len(lt):
            full_seq = int((lt == N_FRAMES_PER_SEQ).sum())
            print(f"  Track lifetimes (n_tracks={len(lt)}):")
            print(f"    mean={lt.mean():.1f}  median={np.median(lt):.0f}  "
                  f"min={lt.min()}  max={lt.max()}")
            print(f"    P(1 frame)  = {(lt==1).mean()*100:.1f}%")
            print(f"    P(≥5 frames)= {(lt>=5).mean()*100:.1f}%")
            print(f"    P(≥10 frames)={(lt>=10).mean()*100:.1f}%")
            print(f"    P(=20 frames)= {(lt==N_FRAMES_PER_SEQ).mean()*100:.1f}%  "
                  f"({full_seq}/{len(lt)} tracks survive full seq)")
        else:
            print("  No tracks found!")

        # 2. Simultaneous targets
        sim = simultaneous_target_stats(h5path)
        p0 = (sim == 0).mean() * 100
        p1 = (sim == 1).mean() * 100
        p2p = (sim >= 2).mean() * 100
        p3p = (sim >= 3).mean() * 100
        print(f"  Simultaneous targets/frame: mean={sim.mean():.2f}  "
              f"max={sim.max()}")
        print(f"    P(0)={p0:.1f}%  P(1)={p1:.1f}%  P(≥2)={p2p:.1f}%  P(≥3)={p3p:.1f}%")

        # 3. Detection feature summary (key fields only)
        det_st, ctx_st = feature_stats(h5path)
        print(f"  Detection features (N_valid={sum(1 for _ in det_st)}  features):")
        print(f"    R_m:   mean={det_st['R_m']['mean']:.1f}m  "
              f"[{det_st['R_m']['p5']:.0f}, {det_st['R_m']['p95']:.0f}] p5-p95")
        print(f"    vr:    mean={det_st['vr_mps']['mean']:+.2f}  "
              f"std={det_st['vr_mps']['std']:.2f} m/s")
        print(f"    az:    mean={det_st['az_deg']['mean']:+.1f}°  "
              f"[{det_st['az_deg']['p5']:.1f}, {det_st['az_deg']['p95']:.1f}] p5-p95")
        print(f"    el:    mean={det_st['el_deg']['mean']:.1f}°  "
              f"[{det_st['el_deg']['p5']:.1f}, {det_st['el_deg']['p95']:.1f}] p5-p95")
        print(f"    Q_dB:  mean={det_st['Q_dB']['mean']:.1f}  "
              f"[{det_st['Q_dB']['p5']:.1f}, {det_st['Q_dB']['p95']:.1f}] p5-p95")

        # 4. Config transition rate
        rate = cfg_transition_rate(h5path)
        print(f"  Config self-transition rate: {rate:.3f}  (target ≈ 0.85)")

    print("\n" + "="*70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp/pilot_diag')
    args = ap.parse_args()

    print(f"Generating {N_PILOT}-sequence pilot for each split → {args.out_dir}")
    paths = generate_pilot(args.out_dir)
    print_report(paths)


if __name__ == '__main__':
    main()
