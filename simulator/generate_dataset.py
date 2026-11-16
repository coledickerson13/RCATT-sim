"""
simulator/generate_dataset.py
==============================
Generate RCATT simulation datasets (train / val / test / ood) and save as HDF5.

HDF5 layout per file  (e.g. data/train.h5):
  contexts    (N_seq, T, 5)          float32   c_k = [B, N_pil, T_obs_ms, P_tx_dBm, P̂_SI_dBm]
  detections  (N_seq, T, M_max, 8)   float32   d_m; ≤1 hit per target per scan (best-beam); padded rows zero
  det_valid   (N_seq, T, M_max)      bool      True where slot holds a real detection
  labels      (N_seq, T, M_max)      int32     0=clutter, >0=target_id, -1=pad
  states      (N_seq, T, N_max, 6)   float32   ground-truth [x,y,z,vx,vy,vz]; padded rows are zero
  state_valid (N_seq, T, N_max)      bool      True where slot holds a live target
  visible     (N_seq, T, N_max)      bool      True where target is inside the surveillance volume
  state_ids   (N_seq, T, N_max)      int32     persistent target ID; −1 = padded slot
  cfg_keys    (N_seq, T)             uint8     index into CONFIG_KEY_LIST
  sic_dbs     (N_seq, T)             float32   SIC suppression in dB (isolation, not residual power)

Attributes on each file:
  config_key_list   list[str]   e.g. ['C1','C2','C3','C4','C5','C6','C7','C8']
  M_max, N_max, T   int
  split             str
  n_sequences       int
  generated_at      str  (ISO timestamp)

Usage
-----
  # pilot (50 sequences per split, 4 workers):
  python -m simulator.generate_dataset --out-dir data/ --pilot 50 --n-workers 4

  # full run:
  python -m simulator.generate_dataset --out-dir data/ --n-workers 8
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Config — imported at module level so workers can re-import cleanly
# ---------------------------------------------------------------------------
from config.sim_config import (
    N_TRAIN, N_VAL, N_TEST, N_OOD,
    N_FRAMES_PER_SEQ, N_TARGET_MAX,
    RESOURCE_CONFIGS,
)
from simulator.scene import run_scene

TRAIN_CONFIGS = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'train']
OOD_CONFIGS   = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'ood']

CONFIG_KEY_LIST = sorted(RESOURCE_CONFIGS.keys())          # ['C1'..'C8']
CONFIG_KEY_TO_IDX = {k: i for i, k in enumerate(CONFIG_KEY_LIST)}

M_MAX = 40   # detection-slot budget (≤5 true detections + Poisson clutter; 40 gives comfortable headroom)
             # + Poisson(10) clutter; 99.99th-pctile total ≈ 28 → 40 gives headroom
N_MAX = N_TARGET_MAX  # = 5

# ---------------------------------------------------------------------------
# Split definitions
# ---------------------------------------------------------------------------
SPLITS = {
    'train': {'n': N_TRAIN, 'cfg_pool': TRAIN_CONFIGS},
    'val':   {'n': N_VAL,   'cfg_pool': TRAIN_CONFIGS},
    'test':  {'n': N_TEST,  'cfg_pool': TRAIN_CONFIGS},
    'ood':   {'n': N_OOD,   'cfg_pool': OOD_CONFIGS},
}


# ---------------------------------------------------------------------------
# Worker  (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

# Config Markov chain (§13: p_stay=0.85, uniform off-diagonal)

def _markov_cfg_sequence(
    cfg_pool: list,
    n_frames: int,
    rng,
    p_stay: float = 0.85,
) -> list:
    """Generate a Markov-chain sequence of config keys.

    Self-transition probability p_stay=0.85 per §13 notes.
    Off-diagonal entries are uniform: (1 - p_stay) / (n_configs - 1).
    For n_configs==1 the chain trivially stays in that config.
    """
    keys = list(cfg_pool)
    n = len(keys)
    cur = int(rng.integers(0, n))
    seq = [keys[cur]]
    for _ in range(n_frames - 1):
        if n == 1 or rng.random() < p_stay:
            seq.append(keys[cur])
        else:
            others = [i for i in range(n) if i != cur]
            cur = int(rng.choice(others))
            seq.append(keys[cur])
    return seq


def _generate_one(seq_idx: int, seed: int, cfg_pool: list, n_frames: int) -> dict:
    """
    Generate one sequence.  Returns a dict of packed numpy arrays.
    Called in a subprocess — must not share state with the parent.
    """
    rng = np.random.default_rng(seed)

    # Random per-frame config selection (not cyclic) from the pool
    # Markov-chain per-frame config selection (p_stay=0.85, §13)
    frame_cfg_keys = _markov_cfg_sequence(cfg_pool, n_frames, rng)

    frames = run_scene(frame_cfg_keys, n_frames, rng)

    # --- Pack into fixed-size arrays ---
    contexts    = np.zeros((n_frames, 5),            dtype=np.float32)
    detections  = np.zeros((n_frames, M_MAX, 8),     dtype=np.float32)
    det_valid   = np.zeros((n_frames, M_MAX),        dtype=bool)
    labels      = np.full( (n_frames, M_MAX), -1,    dtype=np.int32)
    states      = np.zeros((n_frames, N_MAX, 6),     dtype=np.float32)
    state_valid = np.zeros((n_frames, N_MAX),        dtype=bool)
    cfg_keys    = np.zeros( n_frames,                dtype=np.uint8)
    sic_dbs     = np.zeros( n_frames,                dtype=np.float32)
    visible     = np.zeros((n_frames, N_MAX),        dtype=bool)
    state_ids   = np.full( (n_frames, N_MAX), -1,   dtype=np.int32)

    for k, fr in enumerate(frames):
        contexts[k] = fr['context'].astype(np.float32)
        sic_dbs[k]  = np.float32(fr['sic_db'])
        cfg_keys[k] = np.uint8(CONFIG_KEY_TO_IDX[fr['cfg_key']])

        # Detections (variable M)
        dets = fr['detections']   # (M, 8)
        labs = fr['labels']       # (M,)
        M = len(dets)
        if M > M_MAX:
            raise RuntimeError(
                f"Frame {k} has {M} detections, exceeding M_MAX={M_MAX}. "
                "Increase M_MAX or audit the clutter / detection model."
            )
        if M > 0:
            detections[k, :M] = dets[:M].astype(np.float32)
            det_valid[k,  :M] = True
            labels[k,     :M] = labs[:M].astype(np.int32)

        # States (variable N)
        st = fr['states']   # (N, 6)
        N = min(len(st), N_MAX)
        if N > 0:
            states[k,      :N] = st[:N].astype(np.float32)
            state_valid[k, :N] = True

        # Visibility mask and persistent target IDs
        vis = fr.get('visible', np.zeros(N, dtype=bool))
        tids = np.array(fr['ids'], dtype=np.int32)
        Nv = min(len(vis), N_MAX)
        if Nv > 0:
            visible[k,   :Nv] = vis[:Nv]
            state_ids[k, :Nv] = tids[:Nv]

    return {
        'seq_idx':    seq_idx,
        'contexts':   contexts,
        'detections': detections,
        'det_valid':  det_valid,
        'labels':     labels,
        'states':     states,
        'state_valid':state_valid,
        'visible':    visible,
        'state_ids':  state_ids,
        'cfg_keys':   cfg_keys,
        'sic_dbs':    sic_dbs,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def generate_split(
    split_name: str,
    n_sequences: int,
    cfg_pool: list,
    out_path: str,
    base_seed: int,
    n_workers: int,
) -> None:
    """Generate one split and write it to *out_path* (HDF5)."""
    T = N_FRAMES_PER_SEQ

    # Pre-spawn seeds: one per sequence, derived from a SeedSequence for independence
    ss    = np.random.SeedSequence(base_seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_sequences)]

    print(f"\n[{split_name}]  {n_sequences} sequences × {T} frames  "
          f"({len(cfg_pool)} configs: {cfg_pool})  → {out_path}")
    t0 = time.perf_counter()

    # Allocate output arrays up front to avoid realloc
    contexts_all    = np.zeros((n_sequences, T, 5),         dtype=np.float32)
    detections_all  = np.zeros((n_sequences, T, M_MAX, 8),  dtype=np.float32)
    det_valid_all   = np.zeros((n_sequences, T, M_MAX),     dtype=bool)
    labels_all      = np.full( (n_sequences, T, M_MAX), -1, dtype=np.int32)
    states_all      = np.zeros((n_sequences, T, N_MAX, 6),  dtype=np.float32)
    state_valid_all = np.zeros((n_sequences, T, N_MAX),     dtype=bool)
    visible_all     = np.zeros((n_sequences, T, N_MAX),     dtype=bool)
    state_ids_all   = np.full( (n_sequences, T, N_MAX), -1, dtype=np.int32)
    cfg_keys_all    = np.zeros((n_sequences, T),            dtype=np.uint8)
    sic_dbs_all     = np.zeros((n_sequences, T),            dtype=np.float32)

    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_generate_one, i, seeds[i], cfg_pool, T): i
            for i in range(n_sequences)
        }
        for fut in as_completed(futures):
            res = fut.result()
            i   = res['seq_idx']
            contexts_all[i]    = res['contexts']
            detections_all[i]  = res['detections']
            det_valid_all[i]   = res['det_valid']
            labels_all[i]      = res['labels']
            states_all[i]      = res['states']
            state_valid_all[i] = res['state_valid']
            visible_all[i]     = res['visible']
            state_ids_all[i]   = res['state_ids']
            cfg_keys_all[i]    = res['cfg_keys']
            sic_dbs_all[i]     = res['sic_dbs']

            n_done += 1
            if n_done % max(1, n_sequences // 10) == 0 or n_done == n_sequences:
                elapsed = time.perf_counter() - t0
                rate    = n_done / elapsed
                print(f"  {n_done:>{len(str(n_sequences))}}/{n_sequences}  "
                      f"({rate:.1f} seq/s)  elapsed {elapsed:.1f}s")

    # Write HDF5
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('contexts',    data=contexts_all,    compression='gzip', compression_opts=4)
        f.create_dataset('detections',  data=detections_all,  compression='gzip', compression_opts=4)
        f.create_dataset('det_valid',   data=det_valid_all,   compression='gzip', compression_opts=4)
        f.create_dataset('labels',      data=labels_all,      compression='gzip', compression_opts=4)
        f.create_dataset('states',      data=states_all,      compression='gzip', compression_opts=4)
        f.create_dataset('state_valid', data=state_valid_all, compression='gzip', compression_opts=4)
        f.create_dataset('visible',     data=visible_all,     compression='gzip', compression_opts=4)
        f.create_dataset('state_ids',   data=state_ids_all,   compression='gzip', compression_opts=4)
        f.create_dataset('cfg_keys',    data=cfg_keys_all,    compression='gzip', compression_opts=4)
        f.create_dataset('sic_dbs',     data=sic_dbs_all,     compression='gzip', compression_opts=4)

        # Metadata attributes
        f.attrs['split']           = split_name
        f.attrs['n_sequences']     = n_sequences
        f.attrs['T']               = T
        f.attrs['M_max']           = M_MAX
        f.attrs['N_max']           = N_MAX
        f.attrs['config_key_list'] = CONFIG_KEY_LIST
        f.attrs['cfg_pool']        = cfg_pool
        f.attrs['base_seed']       = base_seed
        f.attrs['generated_at']    = datetime.now(timezone.utc).isoformat()

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Done in {elapsed:.1f}s  →  {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Pilot statistics helper
# ---------------------------------------------------------------------------
def print_pilot_stats(h5_path: str) -> None:
    """Print a quick summary of a generated HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        det_valid   = f['det_valid'][:]    # (N, T, M_max)
        labels      = f['labels'][:]       # (N, T, M_max)
        state_valid = f['state_valid'][:]  # (N, T, N_max)
        contexts    = f['contexts'][:]     # (N, T, 5)
        cfg_keys_d  = f['cfg_keys'][:]     # (N, T)
        sic_dbs_d   = f['sic_dbs'][:]     # (N, T)
        cfg_pool    = list(f.attrs['cfg_pool'])
        split       = f.attrs['split']

    M_per_frame  = det_valid.sum(axis=-1)          # (N, T)
    true_mask    = det_valid & (labels > 0)
    clut_mask    = det_valid & (labels == 0)
    true_per_frm = true_mask.sum(axis=-1)          # (N, T)
    clut_per_frm = clut_mask.sum(axis=-1)
    N_per_frame  = state_valid.sum(axis=-1)        # (N, T)

    print(f"\n── Pilot stats: {split} ──────────────────────────")
    print(f"  Sequences:          {det_valid.shape[0]}")
    print(f"  Frames/seq:         {det_valid.shape[1]}")
    print(f"  Mean M/frame:       {M_per_frame.mean():.2f}  "
          f"(std {M_per_frame.std():.2f}, max {M_per_frame.max()})")
    print(f"  Mean true/frame:    {true_per_frm.mean():.2f}  "
          f"(std {true_per_frm.std():.2f})")
    print(f"  Mean clutter/frame: {clut_per_frm.mean():.2f}  "
          f"(std {clut_per_frm.std():.2f})")
    print(f"  Mean N_tgt/frame:   {N_per_frame.mean():.2f}  "
          f"(std {N_per_frame.std():.2f}, max {N_per_frame.max()})")
    print(f"  SIC dB range:       {sic_dbs_d.min():.0f} – {sic_dbs_d.max():.0f}")
    print(f"  Config pool:        {cfg_pool}")
    # Context ranges
    ctx_labels = ['B_PRS_MHz', 'N_pil', 'T_obs_ms', 'P_tx_dBm', 'P_SI_dBm']
    for j, lbl in enumerate(ctx_labels):
        c = contexts[:, :, j]
        print(f"  context[{j}] {lbl:<12s}: "
              f"min={c.min():.2f}  max={c.max():.2f}  mean={c.mean():.2f}")
    # M_max headroom check
    overflow = (M_per_frame >= M_MAX).sum()
    if overflow:
        print(f"  *** WARNING: {overflow} frames hit M_MAX={M_MAX} — increase M_MAX ***")
    else:
        print(f"  M_max headroom OK (no frame hit {M_MAX})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate RCATT HDF5 datasets.")
    p.add_argument('--out-dir',   default='data',  help='Output directory (default: data/)')
    p.add_argument('--n-workers', default=max(1, os.cpu_count() - 1), type=int,
                   help='Parallel workers (default: n_cpu - 1)')
    p.add_argument('--pilot',     default=None, type=int,
                   help='If set, generate only this many sequences per split (quick sanity check)')
    p.add_argument('--seed',      default=0,    type=int, help='Global base seed (default: 0)')
    p.add_argument('--splits',    default='train,val,test,ood',
                   help='Comma-separated list of splits to generate (default: all)')
    return p.parse_args()


def main():
    args   = parse_args()
    splits = [s.strip() for s in args.splits.split(',')]

    print(f"RCATT dataset generator")
    print(f"  out-dir:   {args.out_dir}")
    print(f"  n-workers: {args.n_workers}")
    print(f"  seed:      {args.seed}")
    print(f"  pilot:     {args.pilot if args.pilot else 'full run'}")
    print(f"  splits:    {splits}")

    # Unique base seed per split so sequences are independent across splits
    split_seed_offsets = {'train': 0, 'val': 1, 'test': 2, 'ood': 3}

    for split_name in splits:
        if split_name not in SPLITS:
            print(f"Unknown split '{split_name}', skipping.")
            continue

        cfg         = SPLITS[split_name]
        n_seq       = args.pilot if args.pilot else cfg['n']
        cfg_pool    = cfg['cfg_pool']
        base_seed   = args.seed * 10 + split_seed_offsets.get(split_name, 9)
        suffix      = '_pilot' if args.pilot else ''
        out_path    = os.path.join(args.out_dir, f"{split_name}{suffix}.h5")

        generate_split(
            split_name  = split_name,
            n_sequences = n_seq,
            cfg_pool    = cfg_pool,
            out_path    = out_path,
            base_seed   = base_seed,
            n_workers   = args.n_workers,
        )

        if args.pilot:
            print_pilot_stats(out_path)

    print("All done.")


if __name__ == '__main__':
    main()
