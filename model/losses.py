"""
losses.py
=========
Hungarian-matched tracking loss.

Supervision mask:
  target_mask = state_valid & visible
  — only supervise UAVs that physically exist AND are inside the surveillance
    volume at the reporting boundary.

Default (state_ids=None) — per-frame Hungarian:
  Per (batch, frame): gather GT, build L1 cost matrix, Hungarian assign,
    matched → state L1 + BCE exist=1; unmatched → BCE exist=0.

Identity-consistent (state_ids provided):
  Maintains a per-sequence id_to_query / query_to_id mapping across frames.
  Phase 1 (honour existing): supervised targets already registered → forced pairs.
  Phase 2 (new targets): remaining targets → Hungarian over free queries; register.
  Release: a query is freed when its target's state_id no longer appears in
  state_ids[b,t] (target has left the scene; state_valid → False).

Performance note:
  All matching is performed on CPU tensors to avoid per-iteration CUDA syncs.
  Matched indices are collected across the full batch, then a single vectorized
  GPU gather computes the state L1 loss.

Total loss:
  L = λ_state * mean_L1_per_dim + λ_exist * mean_BCE_per_query
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_loss(
    pred_states       : torch.Tensor,   # (B, T, N_MAX, 6) normalised
    pred_exist_logits : torch.Tensor,   # (B, T, N_MAX)
    gt_states_norm    : torch.Tensor,   # (B, T, N_MAX, 6) normalised GT
    target_mask       : torch.Tensor,   # (B, T, N_MAX) bool: state_valid & visible
    state_ids         = None,           # (B, T, N_MAX) int32; enables identity-consistent matching
    lambda_state : float = 2.0,
    lambda_exist : float = 1.0,
):
    if state_ids is not None:
        return _compute_loss_id(
            pred_states, pred_exist_logits, gt_states_norm, target_mask,
            state_ids, lambda_state, lambda_exist,
        )

    # ── Per-frame Hungarian (baseline, backward-compatible) ──────────────────
    B, T, N_MAX = pred_exist_logits.shape
    device = pred_states.device

    # Pre-move to CPU for matching — eliminates per-iteration CUDA syncs
    pred_cpu = pred_states.detach().cpu()   # (B, T, N_MAX, 6)
    gt_cpu   = gt_states_norm.detach().cpu()

    # Collect all matched pairs; build exist targets on CPU
    matched_b, matched_t, matched_q, matched_g = [], [], [], []
    exist_targets_cpu = torch.zeros(B, T, N_MAX)

    for b in range(B):
        for t in range(T):
            gt_live = target_mask[b, t].nonzero(as_tuple=False).squeeze(1).tolist()
            n_gt = len(gt_live)
            if n_gt == 0:
                continue

            pred_s_cpu = pred_cpu[b, t]           # (N_MAX, 6)
            gt_s_cpu   = gt_cpu[b, t][gt_live]    # (n_gt, 6)

            cost = torch.cdist(
                pred_s_cpu.unsqueeze(0), gt_s_cpu.unsqueeze(0), p=1
            ).squeeze(0) / 6.0                    # (N_MAX, n_gt) — CPU, no sync
            row_idx, col_idx = linear_sum_assignment(cost.numpy())

            for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                matched_b.append(b)
                matched_t.append(t)
                matched_q.append(r)
                matched_g.append(gt_live[c])
                exist_targets_cpu[b, t, r] = 1.0

    n_matched = len(matched_b)
    n_queries = B * T * N_MAX
    exist_targets = exist_targets_cpu.to(device)

    if n_matched > 0:
        bi = torch.tensor(matched_b, dtype=torch.long)
        ti = torch.tensor(matched_t, dtype=torch.long)
        qi = torch.tensor(matched_q, dtype=torch.long)
        gi = torch.tensor(matched_g, dtype=torch.long)
        # Single vectorized GPU gather + L1
        state_loss_sum = F.l1_loss(
            pred_states[bi, ti, qi], gt_states_norm[bi, ti, gi], reduction='sum'
        )
    else:
        state_loss_sum = torch.zeros((), device=device)

    exist_loss_sum = F.binary_cross_entropy_with_logits(
        pred_exist_logits, exist_targets, reduction='sum'
    )

    state_loss = state_loss_sum / (n_matched * 6 + 1e-8)
    exist_loss = exist_loss_sum / (n_queries  + 1e-8)
    loss = lambda_state * state_loss + lambda_exist * exist_loss

    info = {
        "loss"       : loss.item(),
        "state_loss" : state_loss.item(),
        "exist_loss" : exist_loss.item(),
        "n_matched"  : n_matched,
        "n_gt"       : int(target_mask.sum().item()),
        "frac_match" : n_matched / max(int(target_mask.sum().item()), 1),
    }
    return loss, info


# ─────────────────────────────────────────────────────────────────────────────
# Identity-consistent assignment (private helper)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_loss_id(
    pred_states, pred_exist_logits, gt_states_norm, target_mask, state_ids,
    lambda_state, lambda_exist,
):
    """
    Identity-consistent variant — all matching on CPU, one GPU gather at end.

    Key invariant used here:
        state_ids[b, t, n] != -1  ⟺  slot n is state_valid (target alive)
    """
    B, T, N_MAX = pred_exist_logits.shape
    device  = pred_states.device
    ids_cpu = state_ids.cpu()

    # Pre-move to CPU for matching — eliminates per-iteration CUDA syncs
    pred_cpu = pred_states.detach().cpu()     # (B, T, N_MAX, 6)
    gt_cpu   = gt_states_norm.detach().cpu()  # (B, T, N_MAX, 6)
    mask_cpu = target_mask.cpu()              # (B, T, N_MAX)

    # Collect matched pairs across the whole batch
    matched_b, matched_t, matched_q, matched_g = [], [], [], []
    exist_targets_cpu = torch.zeros(B, T, N_MAX)   # built on CPU, moved once

    for b in range(B):
        id_to_query: dict = {}   # state_id  → query_idx
        query_to_id: dict = {}   # query_idx → state_id

        for t in range(T):
            # ── Build alive-ID → gt-slot map ─────────────────────────────────
            alive_id_to_slot = {}
            for n in range(N_MAX):
                sid = int(ids_cpu[b, t, n])
                if sid != -1:
                    alive_id_to_slot[sid] = n

            # ── Release queries whose target has left the scene ───────────────
            for q in list(query_to_id.keys()):
                if query_to_id[q] not in alive_id_to_slot:
                    del id_to_query[query_to_id[q]]
                    del query_to_id[q]

            sup_slots = mask_cpu[b, t].nonzero(as_tuple=False).squeeze(1).tolist()

            # ── Phase 1: honour existing assignments ─────────────────────────
            force_pairs  = []
            new_gt_slots = []

            for g_slot in sup_slots:
                g_id = int(ids_cpu[b, t, g_slot])
                if g_id in id_to_query:
                    force_pairs.append((id_to_query[g_id], g_slot))
                else:
                    new_gt_slots.append((g_id, g_slot))

            # ── Phase 2: Hungarian for new targets on free queries ────────────
            free_queries = [q for q in range(N_MAX) if q not in query_to_id]
            new_pairs    = []

            if new_gt_slots and free_queries:
                new_slot_idxs = [gs for _, gs in new_gt_slots]
                # CPU tensors — no CUDA sync
                pred_free_cpu = pred_cpu[b, t][free_queries]    # (n_free, 6)
                gt_new_cpu    = gt_cpu[b, t][new_slot_idxs]     # (n_new, 6)

                cost = torch.cdist(
                    pred_free_cpu.unsqueeze(0), gt_new_cpu.unsqueeze(0), p=1,
                ).squeeze(0) / 6.0   # (n_free, n_new) — CPU
                row_idx, col_idx = linear_sum_assignment(cost.numpy())

                for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                    q              = free_queries[r]
                    g_id, g_slot   = new_gt_slots[c]
                    new_pairs.append((q, g_slot))
                    id_to_query[g_id] = q
                    query_to_id[q]    = g_id

            # ── Record all matched pairs ──────────────────────────────────────
            for q, g in (force_pairs + new_pairs):
                matched_b.append(b)
                matched_t.append(t)
                matched_q.append(q)
                matched_g.append(g)
                exist_targets_cpu[b, t, q] = 1.0   # CPU scalar write — no sync

    # ── Single vectorized GPU loss computation ────────────────────────────────
    n_matched = len(matched_b)
    n_queries = B * T * N_MAX
    exist_targets = exist_targets_cpu.to(device)   # one H2D transfer

    if n_matched > 0:
        bi = torch.tensor(matched_b, dtype=torch.long)
        ti = torch.tensor(matched_t, dtype=torch.long)
        qi = torch.tensor(matched_q, dtype=torch.long)
        gi = torch.tensor(matched_g, dtype=torch.long)
        # One GPU gather + one L1 call for the entire batch
        state_loss_sum = F.l1_loss(
            pred_states[bi, ti, qi], gt_states_norm[bi, ti, gi], reduction='sum'
        )
    else:
        state_loss_sum = torch.zeros((), device=device)

    exist_loss_sum = F.binary_cross_entropy_with_logits(
        pred_exist_logits, exist_targets, reduction='sum'
    )

    state_loss = state_loss_sum / (n_matched * 6 + 1e-8)
    exist_loss = exist_loss_sum / (n_queries  + 1e-8)
    loss = lambda_state * state_loss + lambda_exist * exist_loss

    info = {
        "loss"       : loss.item(),
        "state_loss" : state_loss.item(),
        "exist_loss" : exist_loss.item(),
        "n_matched"  : n_matched,
        "n_gt"       : int(target_mask.sum().item()),
        "frac_match" : n_matched / max(int(target_mask.sum().item()), 1),
    }
    return loss, info
