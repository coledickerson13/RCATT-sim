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
    """
    Parameters
    ----------
    pred_states, pred_exist_logits : model outputs from AgnosticTransformer.forward()
    gt_states_norm                 : model.norm_states(batch['states'])
    target_mask                    : batch['state_valid'] & batch['visible']
    state_ids                      : batch['state_ids']  (optional; activates id-consistent mode)
    lambda_state, lambda_exist     : loss weights

    Returns
    -------
    loss   : scalar Tensor (requires_grad=True)
    info   : dict of diagnostic floats (detached)
    """
    if state_ids is not None:
        return _compute_loss_id(
            pred_states, pred_exist_logits, gt_states_norm, target_mask,
            state_ids, lambda_state, lambda_exist,
        )

    # ── Per-frame Hungarian (baseline, backward-compatible) ──────────────────
    B, T, N_MAX = pred_exist_logits.shape
    device = pred_states.device

    state_loss_sum = torch.zeros((), device=device)
    exist_loss_sum = torch.zeros((), device=device)
    n_matched      = 0
    n_queries      = 0

    for b in range(B):
        for t in range(T):
            gt_live = target_mask[b, t].nonzero(as_tuple=False).squeeze(1)  # (n_gt,)
            n_gt    = gt_live.numel()

            pred_s = pred_states[b, t]        # (N_MAX, 6)
            pred_l = pred_exist_logits[b, t]  # (N_MAX,)

            exist_tgt = torch.zeros(N_MAX, device=device)

            if n_gt > 0:
                gt_s = gt_states_norm[b, t][gt_live]   # (n_gt, 6)
                cost = torch.cdist(
                    pred_s.unsqueeze(0), gt_s.unsqueeze(0), p=1,
                ).squeeze(0) / 6.0   # (N_MAX, n_gt)
                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
                matched_pred   = pred_s[row_idx]
                matched_gt     = gt_s[col_idx]
                state_loss_sum = state_loss_sum + F.l1_loss(
                    matched_pred, matched_gt, reduction="sum"
                )
                n_matched += len(row_idx)
                exist_tgt[row_idx] = 1.0

            exist_loss_sum = exist_loss_sum + F.binary_cross_entropy_with_logits(
                pred_l, exist_tgt, reduction="sum"
            )
            n_queries += N_MAX

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
    Identity-consistent variant.  See module docstring for algorithm details.

    Key invariant used here:
        state_ids[b, t, n] != -1  ⟺  slot n is state_valid (target alive)
    """
    B, T, N_MAX = pred_exist_logits.shape
    device  = pred_states.device
    ids_cpu = state_ids.cpu()   # small tensor; CPU access is fast

    state_loss_sum = torch.zeros((), device=device)
    exist_loss_sum = torch.zeros((), device=device)
    n_matched      = 0
    n_queries      = 0

    for b in range(B):
        # Per-sequence persistent assignment maps
        id_to_query: dict = {}  # state_id  → query_idx
        query_to_id: dict = {}  # query_idx → state_id

        for t in range(T):
            # ── Build alive-ID → gt-slot map ─────────────────────────────────
            # state_ids[b,t,n] == -1  means slot is padded / target is gone
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

            # ── Supervised GT slots this frame ───────────────────────────────
            sup_slots = target_mask[b, t].nonzero(as_tuple=False).squeeze(1)

            # ── Phase 1: honour existing assignments ─────────────────────────
            force_pairs  = []  # (query_idx, gt_slot)
            new_gt_slots = []  # (state_id, gt_slot) for targets not yet registered

            for g_slot in sup_slots.tolist():
                g_id = int(ids_cpu[b, t, g_slot])
                if g_id in id_to_query:
                    force_pairs.append((id_to_query[g_id], g_slot))
                else:
                    new_gt_slots.append((g_id, g_slot))

            # ── Phase 2: Hungarian for new targets on free queries ────────────
            free_queries = [q for q in range(N_MAX) if q not in query_to_id]
            new_pairs    = []  # (query_idx, gt_slot) freshly assigned

            if new_gt_slots and free_queries:
                new_slot_idxs = [gs for _, gs in new_gt_slots]
                pred_free = pred_states[b, t][
                    torch.tensor(free_queries, dtype=torch.long, device=device)
                ]  # (n_free, 6)
                gt_new = gt_states_norm[b, t][
                    torch.tensor(new_slot_idxs, dtype=torch.long, device=device)
                ]  # (n_new, 6)

                cost = torch.cdist(
                    pred_free.unsqueeze(0), gt_new.unsqueeze(0), p=1,
                ).squeeze(0) / 6.0  # (n_free, n_new)
                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())

                for r, c in zip(row_idx.tolist(), col_idx.tolist()):
                    q              = free_queries[r]
                    g_id, g_slot   = new_gt_slots[c]
                    new_pairs.append((q, g_slot))
                    id_to_query[g_id] = q
                    query_to_id[q]    = g_id

            # ── Compute losses on all matched pairs ───────────────────────────
            all_pairs = force_pairs + new_pairs
            exist_tgt = torch.zeros(N_MAX, device=device)

            if all_pairs:
                q_idxs = torch.tensor([q for q, _ in all_pairs],
                                      dtype=torch.long, device=device)
                g_idxs = torch.tensor([g for _, g in all_pairs],
                                      dtype=torch.long, device=device)
                matched_pred   = pred_states[b, t][q_idxs]       # (n_match, 6)
                matched_gt     = gt_states_norm[b, t][g_idxs]    # (n_match, 6)
                state_loss_sum = state_loss_sum + F.l1_loss(
                    matched_pred, matched_gt, reduction="sum"
                )
                n_matched += len(all_pairs)
                exist_tgt[q_idxs] = 1.0

            exist_loss_sum = exist_loss_sum + F.binary_cross_entropy_with_logits(
                pred_exist_logits[b, t], exist_tgt, reduction="sum"
            )
            n_queries += N_MAX

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
