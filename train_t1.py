"""
train_t1.py
===========
Full T1 (agnostic, no quality, no context) training run.

Trains on data/train.h5 (10 000 sequences), validates on data/val.h5 every
epoch.  Saves the best-validation checkpoint and a full loss CSV.

Hyperparameters
---------------
  d_model=64, n_heads=4, n_enc=2, n_dec=2, dim_ff=128, dropout=0.0
  batch_size=64, lr=5e-4, max_epochs=100, early_stop_patience=15

Usage
-----
  # from rcatt_sim root
  python train_t1.py

  # Colab / custom paths
  python train_t1.py --train data/train.h5 --val data/val.h5 --out out/t1

  # Resume from last checkpoint
  python train_t1.py --resume out/t1/last_model.pt
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.optim as optim
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model import AgnosticTransformer, compute_loss, make_dataloader
from config.sim_config import N_TARGET_MAX


# ── Hyperparameters ──────────────────────────────────────────────────────────
D_MODEL        = 64
N_HEADS        = 4
N_ENC          = 2
N_DEC          = 2
DIM_FF         = 128
DROPOUT        = 0.0     # T1 — no dropout

BATCH_SIZE     = 64
LR             = 5e-4
MAX_EPOCHS     = 100
ES_PATIENCE    = 15      # early stopping: epochs with no val improvement
GRAD_CLIP      = 5.0     # clip_grad_norm_ max_norm

LAMBDA_STATE   = 2.0
LAMBDA_EXIST   = 1.0

LOG_EVERY      = 1       # print every N epochs
CKPT_EVERY     = 10      # save last_model.pt every N epochs (in addition to best)


def set_seed(seed: int = 0):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # allow fast non-deterministic kernels
    torch.backends.cudnn.benchmark     = True    # auto-tune for fixed input sizes


# ── One epoch of training ────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, epoch: int = 0):
    model.train()
    total_loss = state_sum = exist_sum = n_batches = 0

    pbar = tqdm(loader, desc=f"  Ep {epoch:>3d} train", leave=False,
                unit="b", ncols=90, dynamic_ncols=False)
    for batch in pbar:
        detections  = batch['detections' ].to(device, non_blocking=True)
        det_valid   = batch['det_valid'  ].to(device, non_blocking=True)
        states      = batch['states'     ].to(device, non_blocking=True)
        # state_valid, visible, state_ids stay on CPU — losses.py uses them on CPU anyway
        target_mask = batch['state_valid'] & batch['visible']   # (B, T, N_MAX) CPU bool
        state_ids   = batch['state_ids']                        # (B, T, N_MAX) CPU int32

        gt_states_norm = model.norm_states(states)

        pred_states, pred_logits = model(detections, det_valid, contexts=None)

        loss, info = compute_loss(
            pred_states       = pred_states,
            pred_exist_logits = pred_logits,
            gt_states_norm    = gt_states_norm,
            target_mask       = target_mask,
            state_ids         = state_ids,
            lambda_state      = LAMBDA_STATE,
            lambda_exist      = LAMBDA_EXIST,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        state_sum  += info['state_loss']
        exist_sum  += info['exist_loss']
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         st=f"{info['state_loss']:.4f}",
                         ex=f"{info['exist_loss']:.4f}")

    n = max(n_batches, 1)
    return total_loss / n, state_sum / n, exist_sum / n


# ── One epoch of validation ──────────────────────────────────────────────────

def val_epoch(model, loader, device, epoch: int = 0):
    model.eval()
    total_loss = state_sum = exist_sum = n_batches = 0

    pbar = tqdm(loader, desc=f"  Ep {epoch:>3d} val  ", leave=False,
                unit="b", ncols=90, dynamic_ncols=False)
    with torch.no_grad():
        for batch in pbar:
            detections  = batch['detections' ].to(device, non_blocking=True)
            det_valid   = batch['det_valid'  ].to(device, non_blocking=True)
            states      = batch['states'     ].to(device, non_blocking=True)
            target_mask = batch['state_valid'] & batch['visible']   # CPU bool
            state_ids   = batch['state_ids']                        # CPU int32

            gt_states_norm = model.norm_states(states)

            pred_states, pred_logits = model(detections, det_valid, contexts=None)

            loss, info = compute_loss(
                pred_states       = pred_states,
                pred_exist_logits = pred_logits,
                gt_states_norm    = gt_states_norm,
                target_mask       = target_mask,
                state_ids         = state_ids,
                lambda_state      = LAMBDA_STATE,
                lambda_exist      = LAMBDA_EXIST,
            )

            total_loss += loss.item()
            state_sum  += info['state_loss']
            exist_sum  += info['exist_loss']
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = max(n_batches, 1)
    return total_loss / n, state_sum / n, exist_sum / n


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RCATT T1 — Full Training Run")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Train     : {args.train}")
    print(f"  Val       : {args.val}")
    print(f"  Out       : {args.out}")
    print(f"  Batch     : {BATCH_SIZE}  |  LR: {LR}  |  Max epochs: {MAX_EPOCHS}")
    print(f"{'='*60}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5,
    )

    start_epoch = 1
    best_val    = float("inf")
    es_count    = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float("inf"))
        es_count    = ckpt.get('es_count', 0)
        print(f"  Resumed from epoch {ckpt['epoch']}  (best_val={best_val:.4f})\n")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("  Loading datasets into RAM ...", end=" ", flush=True)
    t_load = time.time()
    train_loader = make_dataloader(args.train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = make_dataloader(args.val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"done ({time.time()-t_load:.1f}s)")
    print(f"  Train seqs : {len(train_loader.dataset):,}  ({len(train_loader)} batches/epoch)")
    print(f"  Val seqs   : {len(val_loader.dataset):,}  ({len(val_loader)} batches/epoch)\n")

    # ── CSV header ────────────────────────────────────────────────────────────
    csv_path  = os.path.join(args.out, "loss_history.csv")
    fieldnames = ["epoch", "train_loss", "train_state", "train_exist",
                  "val_loss",   "val_state",   "val_exist",  "lr", "epoch_s"]
    csv_exists = os.path.exists(csv_path) and args.resume
    csv_file   = open(csv_path, "a" if csv_exists else "w", newline="")
    writer     = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    # ── Table header ──────────────────────────────────────────────────────────
    print(f"{'Ep':>4}  {'tr_loss':>8}  {'tr_st':>7}  {'tr_ex':>7}  "
          f"{'vl_loss':>8}  {'vl_st':>7}  {'vl_ex':>7}  {'lr':>8}  {'s/ep':>5}  {'flag'}")
    print("-" * 88)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_st, tr_ex = train_epoch(model, train_loader, optimizer, device, epoch)
        vl_loss, vl_st, vl_ex = val_epoch(model, val_loader, device, epoch)

        scheduler.step(vl_loss)
        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]['lr']

        # ── Checkpoint logic ──────────────────────────────────────────────────
        flag = ""
        if vl_loss < best_val:
            best_val = vl_loss
            es_count = 0
            flag     = "★ best"
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
        else:
            es_count += 1
            if es_count >= ES_PATIENCE:
                flag = "STOP"

        if epoch % CKPT_EVERY == 0 or flag == "STOP":
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val, "es_count": es_count,
            }, os.path.join(args.out, "last_model.pt"))

        # ── Log ───────────────────────────────────────────────────────────────
        writer.writerow({
            "epoch": epoch, "train_loss": f"{tr_loss:.5f}",
            "train_state": f"{tr_st:.5f}", "train_exist": f"{tr_ex:.5f}",
            "val_loss":   f"{vl_loss:.5f}",
            "val_state":  f"{vl_st:.5f}",  "val_exist":  f"{vl_ex:.5f}",
            "lr": f"{cur_lr:.2e}", "epoch_s": f"{elapsed:.1f}",
        })
        csv_file.flush()

        if epoch % LOG_EVERY == 0:
            print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_st:>7.4f}  {tr_ex:>7.4f}  "
                  f"{vl_loss:>8.4f}  {vl_st:>7.4f}  {vl_ex:>7.4f}  "
                  f"{cur_lr:>8.2e}  {elapsed:>5.1f}  {flag}")

        if flag == "STOP":
            print(f"\n  Early stopping triggered (no val improvement for {ES_PATIENCE} epochs).")
            break

    csv_file.close()
    print(f"\n  Best val loss : {best_val:.5f}")
    print(f"  Best model    : {os.path.join(args.out, 'best_model.pt')}")
    print(f"  Loss history  : {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RCATT T1 full training")
    p.add_argument("--train",  default="data/train.h5")
    p.add_argument("--val",    default="data/val.h5")
    p.add_argument("--out",    default="out/t1")
    p.add_argument("--seed",   default=0, type=int)
    p.add_argument("--resume", default=None,
                   help="Path to last_model.pt to resume from")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
