"""
train_t1_profile.py
===================
Full-loop profiler for RCATT T1 training.

Instruments EVERY wall-clock section of the training loop, including the
time the main thread spends blocked in next(train_loader), so the missing
gap between the GPU-only profiler (~570ms) and tqdm (~9.5s) is exposed.

Each section is wrapped with torch.cuda.synchronize() so GPU kernels are
forced to complete before the clock starts or stops.

Usage (Colab):
  # from rcatt_sim root
  !python train_t1_profile.py
  !python train_t1_profile.py --train data/train.h5 --val data/val.h5
"""

import argparse, os, sys, time, collections
import torch
import torch.optim as optim
sys.path.insert(0, os.path.dirname(__file__))
from model import AgnosticTransformer, compute_loss, make_dataloader
from config.sim_config import N_TARGET_MAX

# ── Hyperparameters (must match train_t1.py exactly) ─────────────────────────
D_MODEL, N_HEADS, N_ENC, N_DEC, DIM_FF, DROPOUT = 64, 4, 2, 2, 128, 0.0
BATCH_SIZE   = 64
LR           = 5e-4
LAMBDA_STATE = 2.0
LAMBDA_EXIST = 1.0

N_WARMUP  = 3    # batches discarded before timing starts
N_PROFILE = 10   # batches timed

SECTIONS = [
    "dataloader_wait",   # time blocked in next(iter) — the main suspect
    "h2d_transfer",      # .to(device) for all tensors
    "cpu_preprocess",    # target_mask, state_ids, norm_states (CPU)
    "forward",           # model(detections, det_valid, contexts=None)
    "loss_hungarian",    # compute_loss (cdist + scipy Hungarian, fp32)
    "backward_bptt",     # scaler.scale(loss).backward()
    "clip_optimizer",    # unscale + clip_grad_norm_ + scaler.step/update
    "tqdm_update",       # pbar.set_postfix + pbar.update (pure Python)
    "TOTAL",             # end-to-end wall clock per batch
]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_profile(args):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RCATT T1 — Full-Loop Profiler")
    print(f"{'='*60}")
    print(f"  Device   : {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))
    print(f"  Warmup   : {N_WARMUP} batches   Profile: {N_PROFILE} batches")
    print(f"{'='*60}\n")

    # ── Model + optimizer + scaler ───────────────────────────────────────────
    model = AgnosticTransformer(
        N_MAX=N_TARGET_MAX, d_model=D_MODEL, n_heads=N_HEADS,
        n_enc_layers=N_ENC, n_dec_layers=N_DEC, dim_ff=DIM_FF,
        dropout=DROPOUT, use_quality=False, use_context=False,
    ).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── DataLoader ───────────────────────────────────────────────────────────
    print("  Loading dataset into RAM ...", end=" ", flush=True)
    t0 = time.perf_counter()
    loader = make_dataloader(args.train, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=0)
    print(f"done ({time.perf_counter()-t0:.1f}s)")
    print(f"  {len(loader.dataset):,} sequences  |  {len(loader)} batches/epoch\n")

    # ── Profiling loop ───────────────────────────────────────────────────────
    accum = collections.defaultdict(float)
    batch_num = 0

    # Build a persistent iterator so we measure next() on every batch,
    # not just the first one (re-creating the iterator resets the shuffle).
    data_iter = iter(loader)

    print(f"  {'Batch':>5}  {'dataloader':>12}  {'h2d':>7}  {'cpu_pre':>7}  "
          f"{'forward':>9}  {'loss':>9}  {'backward':>10}  "
          f"{'optimizer':>10}  {'tqdm':>7}  {'TOTAL':>8}")
    print("  " + "-"*105)

    while batch_num < N_WARMUP + N_PROFILE:
        # ── 0. DataLoader wait (CPU-bound, GPU idle) ─────────────────────────
        sync()
        t_batch_start = time.perf_counter()
        t0 = time.perf_counter()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        sync()
        t_dl = time.perf_counter() - t0

        # ── 1. H2D transfer ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        detections = batch['detections'].to(device, non_blocking=True)
        det_valid  = batch['det_valid' ].to(device, non_blocking=True)
        states     = batch['states'    ].to(device, non_blocking=True)
        sync()
        t_h2d = time.perf_counter() - t0

        # ── 2. CPU preprocessing (stays on CPU) ──────────────────────────────
        t0 = time.perf_counter()
        target_mask    = batch['state_valid'] & batch['visible']
        state_ids      = batch['state_ids']
        gt_states_norm = model.norm_states(states)
        sync()
        t_pre = time.perf_counter() - t0

        # ── 3. Forward pass (fp16 via autocast) ──────────────────────────────
        sync()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred_states, pred_logits = model(detections, det_valid, contexts=None)
        sync()
        t_fwd = time.perf_counter() - t0

        # ── 4. Loss / Hungarian (fp32 — cdist doesn't support Half) ──────────
        sync()
        t0 = time.perf_counter()
        loss, info = compute_loss(
            pred_states       = pred_states.float(),
            pred_exist_logits = pred_logits.float(),
            gt_states_norm    = gt_states_norm,
            target_mask       = target_mask,
            state_ids         = state_ids,
            lambda_state      = LAMBDA_STATE,
            lambda_exist      = LAMBDA_EXIST,
        )
        sync()
        t_loss = time.perf_counter() - t0

        # ── 5. Backward (BPTT through GRU) ───────────────────────────────────
        optimizer.zero_grad()
        sync()
        t0 = time.perf_counter()
        scaler.scale(loss).backward()
        sync()
        t_bwd = time.perf_counter() - t0

        # ── 6. Clip + optimizer step ─────────────────────────────────────────
        sync()
        t0 = time.perf_counter()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        sync()
        t_opt = time.perf_counter() - t0

        # ── 7. tqdm-equivalent overhead (Python-only) ────────────────────────
        t0 = time.perf_counter()
        loss_val  = loss.item()           # scalar D2H — included here
        _st = info['state_loss']
        _ex = info['exist_loss']
        # simulate set_postfix string formatting
        _ = f"loss={loss_val:.4f} st={_st:.4f} ex={_ex:.4f}"
        t_tqdm = time.perf_counter() - t0

        sync()
        t_total = time.perf_counter() - t_batch_start

        batch_num += 1
        if batch_num <= N_WARMUP:
            print(f"  {'(warmup)':>5}")
            continue

        # Accumulate
        accum["dataloader_wait"] += t_dl
        accum["h2d_transfer"]    += t_h2d
        accum["cpu_preprocess"]  += t_pre
        accum["forward"]         += t_fwd
        accum["loss_hungarian"]  += t_loss
        accum["backward_bptt"]   += t_bwd
        accum["clip_optimizer"]  += t_opt
        accum["tqdm_update"]     += t_tqdm
        accum["TOTAL"]           += t_total

        n = batch_num - N_WARMUP
        print(f"  {n:>5}  "
              f"{t_dl*1e3:>10.1f}ms  "
              f"{t_h2d*1e3:>5.1f}ms  "
              f"{t_pre*1e3:>5.1f}ms  "
              f"{t_fwd*1e3:>7.1f}ms  "
              f"{t_loss*1e3:>7.1f}ms  "
              f"{t_bwd*1e3:>8.1f}ms  "
              f"{t_opt*1e3:>8.1f}ms  "
              f"{t_tqdm*1e3:>5.1f}ms  "
              f"{t_total*1e3:>6.1f}ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = N_PROFILE
    print(f"\n  {'='*60}")
    print(f"  AVERAGES over {n} batches  (batch_size={BATCH_SIZE})")
    print(f"  {'='*60}")
    total_avg = accum['TOTAL'] / n * 1e3
    for key in SECTIONS:
        avg_ms  = accum[key] / n * 1e3
        pct     = avg_ms / total_avg * 100 if total_avg > 0 else 0
        bar     = '█' * int(pct / 2)
        print(f"  {key:<22}  {avg_ms:>8.1f} ms   {pct:>5.1f}%  {bar}")

    epoch_s = accum['TOTAL'] / n * len(loader)
    print(f"\n  Projected epoch time : {epoch_s:.0f}s  ({epoch_s/60:.1f} min)")
    print(f"  ({len(loader)} batches/epoch × {total_avg:.0f}ms/batch)\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.h5")
    return p.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
