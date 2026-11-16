"""
train_overfit.py
================
Staged overfitting check for the T1 AgnosticTransformer.

A well-designed model should memorise a small fixed dataset.  Run in three
stages (GPT review Nov-16 recommendation):

  Stage 1 — 1 sequence,  ≤300 epochs   isolates optimisation problems
  Stage 2 — 4 sequences, ≤150 epochs   confirms multi-scene learning
  Stage 3 — 20 sequences, ≤100 epochs  capacity & identity stability check

Metrics logged every LOG_INTERVAL epochs
-----------------------------------------
  state_loss  — mean-L1 (normalised) on matched pairs          (from loss fn)
  exist_loss  — mean-BCE on all queries                        (from loss fn)
  pos_rmse_m  — position RMSE in metres  (Hungarian, L2 cost)
  vel_rmse_mps — velocity RMSE in m/s
  exist_prec  — existence precision  (threshold σ(logit) > 0.5)
  exist_rec   — existence recall
  count_mae   — mean |predicted count − GT count| per frame

Outputs (--out-dir, default: out/overfit/)
------------------------------------------
  best_model.pt      — state-dict at minimum training loss
  loss_history.csv   — per-epoch scalars

Verdict thresholds (advisory — not architectural failure criteria)
------------------------------------------------------------------
  ✓ PASS    — final loss < 0.05
  ⚠ PARTIAL — final loss 0.05–0.15  (more epochs / capacity may converge)
  ✗ FAIL    — final loss > 0.15     (exits with code 1 for CI use)

Usage
-----
  python train_overfit.py                          # stage 3: 20 seqs, 100 epochs
  python train_overfit.py --n-seqs 1  --epochs 300 # stage 1
  python train_overfit.py --n-seqs 4  --epochs 150 # stage 2
  python train_overfit.py --pilot-dir data/ --seed 0
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))

from model import AgnosticTransformer, compute_loss, make_dataloader
from config.sim_config import N_TARGET_MAX, N_FRAMES_PER_SEQ


# ── Hyperparameters ──────────────────────────────────────────────────────────
N_SEQS_DEFAULT = 20
EPOCHS_DEFAULT = 100
LR             = 5e-4
D_MODEL        = 64
N_HEADS        = 4
N_ENC          = 2
N_DEC          = 2
DIM_FF         = 128
DROPOUT        = 0.0   # OFF — we want to overfit

LOG_INTERVAL   = 10    # print + compute eval metrics every N epochs
STALL_PATIENCE = 25    # consecutive epochs without ≥1% improvement → stall warn

# Physical-unit scales (must match AgnosticTransformer._STATE_SCALE)
_POS_SCALE = 300.0  # metres
_VEL_SCALE  = 25.0  # m s⁻¹


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix torch, numpy, and Python RNG for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Evaluation metrics ────────────────────────────────────────────────────────

def compute_eval_metrics(
    pred_states       : torch.Tensor,   # (B, T, N_MAX, 6) normalised
    pred_exist_logits : torch.Tensor,   # (B, T, N_MAX)
    gt_states_phys    : torch.Tensor,   # (B, T, N_MAX, 6) physical (m, m/s)
    target_mask       : torch.Tensor,   # (B, T, N_MAX) bool
) -> dict:
    """
    Compute physical-unit tracking quality metrics.

    Existence P/R and state RMSE are both derived from the *same* per-frame
    Hungarian assignment between the N_MAX predicted queries and the n_gt live
    GT targets.  This is correct regardless of which query index the loss
    function matched to which GT slot — a query tracking GT slot 4 will still
    be matched to that target here via position cost.

    Returns
    -------
    pos_rmse_m    : 3-D Euclidean position RMSE [m]
    vel_rmse_mps  : 3-D Euclidean velocity RMSE [m/s]
    exist_prec    : precision  — matched predicted tracks / all σ>0.5 predictions
    exist_rec     : recall     — matched GT targets       / all live GT targets
    count_mae     : mean |predicted_count − gt_count| per (batch, frame)
    """
    with torch.no_grad():
        # Convert normalised predictions to physical units
        state_scale = pred_states.new_tensor(
            [_POS_SCALE, _POS_SCALE, _POS_SCALE,
             _VEL_SCALE, _VEL_SCALE, _VEL_SCALE]
        )
        pred_phys  = pred_states * state_scale   # (B, T, N, 6)
        pred_exist = torch.sigmoid(pred_exist_logits) > 0.5   # (B, T, N) bool

        # ── Count MAE (slot-free — just totals) ───────────────────────────────
        pred_count = pred_exist.float().sum(dim=-1)   # (B, T)
        gt_count   = target_mask.float().sum(dim=-1)  # (B, T)
        count_mae  = (pred_count - gt_count).abs().mean().item()

        # ── Per-frame Hungarian: same assignment for RMSE and P/R ─────────────
        B, T, N, _ = pred_phys.shape
        pos_sq_all = []   # squared 3-D position errors
        vel_sq_all = []   # squared 3-D velocity errors
        TP = FP = FN = 0

        for b in range(B):
            for t in range(T):
                gt_live  = target_mask[b, t].nonzero(as_tuple=False).squeeze(1)
                n_gt     = gt_live.numel()
                pred_pos = pred_phys[b, t, :, :3]       # (N, 3)
                gt_pos   = gt_states_phys[b, t][gt_live, :3]  # (n_gt, 3)

                # Total predicted tracks this frame
                n_pred = pred_exist[b, t].sum().item()

                if n_gt == 0:
                    FP += int(n_pred)
                    continue

                if n_pred == 0:
                    FN += n_gt
                    continue

                # Hungarian on 3-D position cost (all N queries vs n_gt GT)
                cost     = torch.cdist(
                    pred_pos.unsqueeze(0), gt_pos.unsqueeze(0), p=2,
                ).squeeze(0)                              # (N, n_gt)
                row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())

                # A matched pair is a TP when the assigned query predicts "exist"
                for qi, gi in zip(row_idx, col_idx):
                    if pred_exist[b, t, qi]:
                        TP  += 1
                        diff = pred_phys[b, t, qi] - gt_states_phys[b, t, gt_live[gi]]
                        pos_sq_all.append((diff[:3] ** 2).sum().item())
                        vel_sq_all.append((diff[3:] ** 2).sum().item())
                    else:
                        FN += 1   # GT matched but query says "absent"

                # Unmatched queries that still fire → FP
                matched_queries = set(row_idx.tolist())
                for qi in range(N):
                    if pred_exist[b, t, qi] and qi not in matched_queries:
                        FP += 1

        exist_prec = TP / (TP + FP + 1e-8)
        exist_rec  = TP / (TP + FN + 1e-8)

        # 3-D Euclidean RMSE: sqrt(mean of sum-of-squares per point)
        pos_rmse = math.sqrt(sum(pos_sq_all) / len(pos_sq_all)) if pos_sq_all else 0.0
        vel_rmse = math.sqrt(sum(vel_sq_all) / len(vel_sq_all)) if vel_sq_all else 0.0

    return {
        "pos_rmse_m"  : pos_rmse,
        "vel_rmse_mps": vel_rmse,
        "exist_prec"  : exist_prec,
        "exist_rec"   : exist_rec,
        "count_mae"   : count_mae,
    }


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_pilot(out_dir: str, n_seqs: int) -> str:
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "train_pilot.h5")
    if os.path.exists(h5_path):
        print(f"[overfit] Reusing existing pilot: {h5_path}")
        return h5_path
    print(f"[overfit] Generating {n_seqs}-sequence pilot → {h5_path} …")
    cmd = [
        sys.executable, "-m", "simulator.generate_dataset",
        "--out-dir", out_dir,
        "--pilot", str(n_seqs),
        "--splits", "train",
        "--n-workers", "1",
    ]
    subprocess.run(cmd, check=True)
    return h5_path


# ── Training loop ─────────────────────────────────────────────────────────────

def run_overfit(h5_path: str, n_epochs: int, out_dir: str, seed: int,
                n_seqs_expected: int = 0, lr: float = LR) -> float:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[overfit] Device:  {device}")
    print(f"[overfit] Data:    {h5_path}")
    print(f"[overfit] Epochs:  {n_epochs}  |  seed: {seed}")
    print(f"[overfit] Out:     {out_dir}")

    # ── Model (T1: no quality, no context, no dropout) ────────────────────────
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
    print(f"[overfit] Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5,
    )

    # batch_size=256 → single batch for any pilot ≤256 seqs
    loader = make_dataloader(h5_path, batch_size=256, shuffle=False, num_workers=0)
    n_dataset = len(loader.dataset)
    if n_seqs_expected and n_dataset != n_seqs_expected:
        raise RuntimeError(
            f"[overfit] Dataset size mismatch: expected {n_seqs_expected} sequences "
            f"but {h5_path} contains {n_dataset}.  Delete the file and re-run to "
            f"regenerate with the correct size."
        )
    print(f"[overfit] Dataset:  {n_dataset} sequences "
          f"({math.ceil(n_dataset / 256)} batch/epoch)\n")

    best_loss    = float("inf")
    stall_count  = 0
    epoch_losses = []
    history      = []   # rows for CSV

    # Header
    print(
        f"{'Epoch':>6}  {'loss':>7}  {'state':>7}  {'exist':>7}  "
        f"{'pos_m':>7}  {'vel_mps':>7}  {'prec':>5}  {'rec':>5}  {'cnt_mae':>7}"
    )
    print("-" * 76)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches    = 0
        last_info    = {}

        for batch in loader:
            detections  = batch['detections' ].to(device)
            det_valid   = batch['det_valid'  ].to(device)
            states      = batch['states'     ].to(device)
            state_valid = batch['state_valid'].to(device)
            visible     = batch['visible'    ].to(device)
            state_ids   = batch['state_ids'  ].to(device)

            target_mask    = state_valid & visible
            gt_states_norm = model.norm_states(states)

            pred_states, pred_logits = model(detections, det_valid, contexts=None)

            loss, info = compute_loss(
                pred_states       = pred_states,
                pred_exist_logits = pred_logits,
                gt_states_norm    = gt_states_norm,
                target_mask       = target_mask,
                state_ids         = state_ids,
            )

            assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch}"

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent BCE explosion
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1
            last_info     = info

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        # ── LR schedule step ──────────────────────────────────────────────────
        scheduler.step(avg_loss)

        # ── Best-model checkpoint (post-update weights, post-update loss) ─────
        if avg_loss < best_loss:
            best_loss   = avg_loss
            stall_count = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            stall_count += 1

        # ── Eval metrics: fresh model.eval() pass over the full dataset ────────
        # Runs after optimizer.step() so predictions reflect the updated weights.
        do_eval = (epoch == 1 or epoch % LOG_INTERVAL == 0 or epoch == n_epochs)
        em = {}
        if do_eval:
            model.eval()
            with torch.no_grad():
                all_ps, all_pl, all_st, all_tm = [], [], [], []
                for batch in loader:
                    detections  = batch['detections' ].to(device)
                    det_valid   = batch['det_valid'  ].to(device)
                    states      = batch['states'     ].to(device)
                    state_valid = batch['state_valid'].to(device)
                    visible     = batch['visible'    ].to(device)
                    ps, pl = model(detections, det_valid, contexts=None)
                    all_ps.append(ps);  all_pl.append(pl)
                    all_st.append(states);  all_tm.append(state_valid & visible)
            model.train()
            em = compute_eval_metrics(
                torch.cat(all_ps), torch.cat(all_pl),
                torch.cat(all_st), torch.cat(all_tm),
            )

        # ── History row ───────────────────────────────────────────────────────
        row = {
            "epoch"      : epoch,
            "loss"       : avg_loss,
            "state_loss" : last_info.get("state_loss", ""),
            "exist_loss" : last_info.get("exist_loss", ""),
            "pos_rmse_m" : em.get("pos_rmse_m",   ""),
            "vel_rmse_mps": em.get("vel_rmse_mps",""),
            "exist_prec" : em.get("exist_prec",   ""),
            "exist_rec"  : em.get("exist_rec",    ""),
            "count_mae"  : em.get("count_mae",    ""),
        }
        history.append(row)

        # ── Print ──────────────────────────────────────────────────────────────
        if do_eval:
            stall_flag = "  ⚠ STALL" if stall_count >= STALL_PATIENCE else ""
            print(
                f"{epoch:>6}  {avg_loss:>7.4f}  "
                f"{last_info.get('state_loss', 0):>7.4f}  "
                f"{last_info.get('exist_loss', 0):>7.4f}  "
                f"{em.get('pos_rmse_m',   0):>7.1f}  "
                f"{em.get('vel_rmse_mps', 0):>7.3f}  "
                f"{em.get('exist_prec',   0):>5.2f}  "
                f"{em.get('exist_rec',    0):>5.2f}  "
                f"{em.get('count_mae',    0):>7.3f}"
                f"{stall_flag}"
            )

        if avg_loss < 0.02:
            print(f"\n[overfit] ✓ Converged at epoch {epoch} — loss {avg_loss:.4f} < 0.02 target.")
            break

    # ── Save CSV history ──────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "loss_history.csv")
    fieldnames = [
        "epoch", "loss", "state_loss", "exist_loss",
        "pos_rmse_m", "vel_rmse_mps", "exist_prec", "exist_rec", "count_mae",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"\n[overfit] Loss history saved → {csv_path}")
    print(f"[overfit] Best model saved  → {os.path.join(out_dir, 'best_model.pt')}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    final = epoch_losses[-1]
    start = epoch_losses[0]
    pct   = (start - final) / (start + 1e-8) * 100

    if final < 0.05:
        verdict  = f"✓ PASS    — loss converged to {final:.4f}  (target ≤ 0.05)"
        exit_ok  = True
    elif final < 0.15:
        verdict  = f"⚠ PARTIAL — loss reached {final:.4f}  (more epochs / capacity may converge)"
        exit_ok  = True
    else:
        verdict  = f"✗ FAIL    — loss only reached {final:.4f}  after {len(epoch_losses)} epochs"
        exit_ok  = False

    print(f"[overfit] {verdict}")
    print(f"[overfit] Loss reduced {pct:.1f}%: {start:.4f} → {final:.4f}")
    print(
        "[overfit] NOTE: thresholds are aspirational targets, not architectural "
        "failure criteria.  A FAIL after 100 epochs warrants more epochs / stages, "
        "not necessarily a redesign."
    )

    if stall_count >= STALL_PATIENCE:
        print(
            f"[overfit] ⚠ Loss stalled for {stall_count} consecutive epochs — "
            "consider increasing d_model / n_layers, or reducing LR."
        )

    return final, exit_ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RCATT overfitting check (T1 baseline)")
    p.add_argument("--pilot-dir", default="data",
                   help="Directory containing train_pilot.h5 (default: data/)")
    p.add_argument("--n-seqs",   default=N_SEQS_DEFAULT, type=int,
                   help=f"Sequences to generate if no pilot found (default: {N_SEQS_DEFAULT})")
    p.add_argument("--epochs",   default=EPOCHS_DEFAULT, type=int,
                   help=f"Max training epochs (default: {EPOCHS_DEFAULT})")
    p.add_argument("--out-dir",  default="out/overfit",
                   help="Directory for best_model.pt and loss_history.csv (default: out/overfit/)")
    p.add_argument("--seed",     default=42, type=int,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--lr",       default=LR, type=float,
                   help=f"Initial Adam LR (default: {LR}); "
                        "ReduceLROnPlateau halves it after 15 stall epochs")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    h5_path = os.path.join(args.pilot_dir, "train_pilot.h5")

    if not os.path.exists(h5_path):
        h5_path = build_pilot(args.pilot_dir, n_seqs=args.n_seqs)

    final_loss, ok = run_overfit(
        h5_path          = h5_path,
        n_epochs         = args.epochs,
        out_dir          = args.out_dir,
        seed             = args.seed,
        n_seqs_expected  = args.n_seqs,
        lr               = args.lr,
    )

    sys.exit(0 if ok else 1)
