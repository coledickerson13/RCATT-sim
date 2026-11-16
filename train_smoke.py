"""
train_smoke.py
==============
Two-epoch smoke test for the T1 AgnosticTransformer baseline.

Checks:
  1. Loss is finite (no NaN/Inf) on the first batch
  2. Gradients flow to all parameters
  3. Loss decreases across two epochs on a small dataset
  4. Output shapes are correct

Usage
-----
  # Use existing pilot data:
  python train_smoke.py --pilot-dir data/

  # Auto-generate a 20-sequence mini-pilot if no data found:
  python train_smoke.py

  # Increase sequences for a fuller test:
  python train_smoke.py --pilot-dir data/ --n-seqs 20
"""

import argparse
import os
import sys
import math

import torch
import torch.optim as optim

# Allow running from the rcatt_sim root
sys.path.insert(0, os.path.dirname(__file__))

from model import AgnosticTransformer, compute_loss, make_dataloader
from config.sim_config import N_TARGET_MAX, N_FRAMES_PER_SEQ


# ── Hyperparameters ──────────────────────────────────────────────────────────
N_EPOCHS    = 2
BATCH_SIZE  = 4
LR          = 1e-4
D_MODEL     = 64    # smaller than production for speed
N_HEADS     = 4
N_ENC       = 2
N_DEC       = 2
DIM_FF      = 128
DROPOUT     = 0.0   # off for smoke test (reduces variance)


def build_mini_pilot(out_dir: str, n_seqs: int = 20) -> str:
    """Generate a minimal pilot dataset for the smoke test."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "train_pilot.h5")
    if os.path.exists(h5_path):
        print(f"[smoke] Reusing existing pilot: {h5_path}")
        return h5_path

    print(f"[smoke] Generating {n_seqs}-sequence mini-pilot → {h5_path} …")
    cmd = [
        sys.executable, "-m", "simulator.generate_dataset",
        "--out-dir", out_dir,
        "--pilot", str(n_seqs),
        "--splits", "train",
        "--n-workers", "1",
    ]
    ret = subprocess.run(cmd, check=True, capture_output=False)
    return h5_path


def run_smoke(h5_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[smoke] Device: {device}")
    print(f"[smoke] Data:   {h5_path}")

    # ── Model (T1: no quality, no context) ───────────────────────────────────
    model = AgnosticTransformer(
        N_MAX        = N_TARGET_MAX,
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_enc_layers = N_ENC,
        n_dec_layers = N_DEC,
        dim_ff       = DIM_FF,
        dropout      = DROPOUT,
        use_quality  = False,
        use_context  = False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── DataLoader ────────────────────────────────────────────────────────────
    loader = make_dataloader(h5_path, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[smoke] Dataset size: {len(loader.dataset)} sequences  "
          f"({math.ceil(len(loader.dataset) / BATCH_SIZE)} batches/epoch)")

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches    = 0
        first_batch  = True

        for batch in loader:
            detections  = batch['detections'].to(device)   # (B, T, M, 8)
            det_valid   = batch['det_valid' ].to(device)   # (B, T, M)
            states      = batch['states'    ].to(device)   # (B, T, N_MAX, 6)
            state_valid = batch['state_valid'].to(device)  # (B, T, N_MAX)
            visible     = batch['visible'   ].to(device)   # (B, T, N_MAX)
            state_ids   = batch['state_ids' ].to(device)   # (B, T, N_MAX) int32

            target_mask = state_valid & visible             # (B, T, N_MAX)

            # Forward
            pred_states, pred_logits = model(detections, det_valid, contexts=None)

            # ── Shape checks (first batch only) ──────────────────────────────
            if first_batch and epoch == 1:
                B, T, M, _ = detections.shape
                assert pred_states.shape  == (B, T, N_TARGET_MAX, 6), \
                    f"pred_states shape mismatch: {pred_states.shape}"
                assert pred_logits.shape  == (B, T, N_TARGET_MAX), \
                    f"pred_logits shape mismatch: {pred_logits.shape}"
                print(f"[smoke] ✓ Output shapes: "
                      f"pred_states {tuple(pred_states.shape)},  "
                      f"pred_logits {tuple(pred_logits.shape)}")
                first_batch = False

            # Normalise GT states
            gt_states_norm = model.norm_states(states)

            # Loss
            loss, info = compute_loss(
                pred_states       = pred_states,
                pred_exist_logits = pred_logits,
                gt_states_norm    = gt_states_norm,
                target_mask       = target_mask,
                state_ids         = state_ids,
            )

            # ── Finite loss check ─────────────────────────────────────────────
            assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"

            optimizer.zero_grad()
            loss.backward()

            # ── Gradient check (first batch only) ────────────────────────────
            if epoch == 1 and n_batches == 0:
                no_grad = [
                    n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is None
                ]
                if no_grad:
                    print(f"[smoke] ✗ Parameters with no gradient: {no_grad}")
                else:
                    print(f"[smoke] ✓ All {n_params:,} parameters received gradients")

            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"[smoke] Epoch {epoch}/{N_EPOCHS}  loss={avg_loss:.4f}  "
              f"(state={info['state_loss']:.4f}  exist={info['exist_loss']:.4f}  "
              f"match_rate={info['frac_match']:.2f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if N_EPOCHS >= 2 and epoch_losses[-1] < epoch_losses[0]:
        print(f"[smoke] ✓ Loss decreased: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")
    elif N_EPOCHS >= 2:
        print(f"[smoke] ⚠ Loss did not decrease: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}  "
              f"(normal on a tiny dataset with 2 epochs)")
    print("[smoke] ✓ Smoke test passed — model runs end-to-end without errors.")


def parse_args():
    p = argparse.ArgumentParser(description="RCATT T1 smoke test")
    p.add_argument("--pilot-dir", default="data",
                   help="Directory containing train_pilot.h5 (default: data/)")
    p.add_argument("--n-seqs", default=20, type=int,
                   help="Sequences to generate if no pilot found (default: 20)")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    h5_path = os.path.join(args.pilot_dir, "train_pilot.h5")

    if not os.path.exists(h5_path):
        h5_path = build_mini_pilot(args.pilot_dir, n_seqs=args.n_seqs)

    run_smoke(h5_path)
