"""
tests/test_identity_loss.py
===========================
Pytest suite for identity-consistent matching in compute_loss.

Run from rcatt_sim/:
    pytest tests/test_identity_loss.py -v

Tests
-----
  test_identity_switch_penalised  — swapping state_ids raises loss vs consistent IDs
  test_reentry_query_preserved    — query stays reserved while target is alive but invisible
  test_death_releases_query       — query freed when state_id disappears (target gone)
  test_all_absent_finite          — no targets → finite loss, gradients flow
  test_gt_slot_reorder_stable     — same IDs packed into different GT slots → same loss
  test_context_guard_raises       — use_context=True + contexts=None → ValueError
"""

import pytest
import torch

from model.losses import compute_loss
from model.agnostic_transformer import AgnosticTransformer


# ── Helpers ───────────────────────────────────────────────────────────────────

N_MAX   = 3
D_STATE = 6


def _ids(B, T, N, assignments):
    """
    Build a state_ids tensor.
    assignments: list of (b, t, n, id) tuples; all other slots default to -1.
    """
    t = torch.full((B, T, N), -1, dtype=torch.int32)
    for b, fr, n, sid in assignments:
        t[b, fr, n] = sid
    return t


def _mask(B, T, N, live):
    """
    Build a target_mask bool tensor.
    live: list of (b, t, n) tuples.
    """
    m = torch.zeros(B, T, N, dtype=torch.bool)
    for b, fr, n in live:
        m[b, fr, n] = True
    return m


# ── Test 1: Identity switch is penalised ─────────────────────────────────────

def test_identity_switch_penalised():
    """
    Two targets in GT: ID=1 always at x=+1, ID=2 always at x=−1.
    Predictions: query 0 near +1, query 1 near −1.

    With consistent IDs the matcher forces each query to its known target →
    state loss ≈ 0.  With swapped IDs at frame 1 the same queries are forced
    to the wrong GTs → higher state loss.
    """
    B, T, N = 1, 3, N_MAX

    gt = torch.zeros(B, T, N, D_STATE)
    gt[0, :, 0, 0] =  1.0   # slot 0: x = +1
    gt[0, :, 1, 0] = -1.0   # slot 1: x = −1

    # Predictions fixed close to their natural GT
    pred_s = torch.zeros(B, T, N, D_STATE)
    pred_s[0, :, 0, 0] =  1.0
    pred_s[0, :, 1, 0] = -1.0
    pred_s = pred_s.detach().requires_grad_(True)
    pred_l = torch.zeros(B, T, N, requires_grad=True)

    mask = _mask(B, T, N, [(0, t, n) for t in range(T) for n in range(2)])

    # Case A — consistent: ID=1 always slot 0, ID=2 always slot 1
    ids_a = _ids(B, T, N, [(0, t, 0, 1) for t in range(T)] +
                           [(0, t, 1, 2) for t in range(T)])

    # Case B — swapped at frame 1: slot 0↔slot 1 swap IDs
    ids_b = ids_a.clone()
    ids_b[0, 1, 0] = 2
    ids_b[0, 1, 1] = 1

    loss_a, _ = compute_loss(pred_s, pred_l, gt, mask, state_ids=ids_a)
    loss_b, _ = compute_loss(pred_s, pred_l, gt, mask, state_ids=ids_b)

    assert torch.isfinite(loss_a), "Consistent-ID loss is non-finite"
    assert torch.isfinite(loss_b), "Swapped-ID loss is non-finite"
    assert loss_b > loss_a + 0.05, (
        f"Expected swapped loss ({loss_b:.4f}) >> consistent loss ({loss_a:.4f})"
    )


# ── Test 2: Reentry — query stays reserved during invisible frame ─────────────

def test_reentry_query_preserved():
    """
    UAV (ID=1) is visible at t=0 and t=2, but NOT at t=1 (outside coverage,
    target_mask=False).  state_ids[0,1,0]=1  — the target is still alive.

    The query assigned at t=0 must NOT be released during t=1 (target alive,
    just invisible).  At t=2 the same query must handle the target again.

    With a perfect prediction (pred ≈ GT) the state loss should stay near 0.
    """
    B, T, N = 1, 3, N_MAX

    gt = torch.zeros(B, T, N, D_STATE)
    gt[0, :, 0, 0] = 1.0        # ID=1 always at x=1

    pred_s = torch.zeros(B, T, N, D_STATE)
    pred_s[0, :, 0, 0] =  1.0   # query 0 tracks at x=1
    pred_s[0, :, 1, 0] = -99.0  # query 1 far away (should not be chosen)
    pred_s = pred_s.detach().requires_grad_(True)
    pred_l = torch.zeros(B, T, N, requires_grad=True)

    # Visible at t=0 and t=2 only (t=1 outside coverage)
    mask = _mask(B, T, N, [(0, 0, 0), (0, 2, 0)])

    # state_ids: ID=1 alive for all T (state_valid=True), slot 0
    ids = _ids(B, T, N, [(0, t, 0, 1) for t in range(T)])

    loss, info = compute_loss(pred_s, pred_l, gt, mask, state_ids=ids)

    assert torch.isfinite(loss), "Loss is non-finite in reentry scenario"
    # Perfect prediction → state loss near 0
    assert info["state_loss"] < 0.1, (
        f"State loss {info['state_loss']:.4f} too high — query may not be preserved"
    )
    assert info["n_matched"] == 2   # t=0 and t=2 only


# ── Test 3: Death releases query ──────────────────────────────────────────────

def test_death_releases_query():
    """
    ID=1 occupies slot 0 at t=0 and t=1, then dies (state_ids[0,2,0]=-1).
    ID=2 enters slot 0 at t=2.

    The query that served ID=1 must be released so ID=2 can be assigned to it.
    We expect 3 matched pairs (one per frame, one live target each).
    """
    B, T, N = 1, 3, N_MAX

    gt = torch.zeros(B, T, N, D_STATE)
    gt[0, 0, 0, 0] = 1.0   # ID=1 at t=0
    gt[0, 1, 0, 0] = 1.0   # ID=1 at t=1
    gt[0, 2, 0, 0] = 2.0   # ID=2 at t=2 (slot reused)

    pred_s = torch.zeros(B, T, N, D_STATE)
    pred_s[0, :, 0, 0] = 1.5   # query 0 somewhere between 1 and 2
    pred_s = pred_s.detach().requires_grad_(True)
    pred_l = torch.zeros(B, T, N, requires_grad=True)

    # All three frames have one visible target
    mask = _mask(B, T, N, [(0, 0, 0), (0, 1, 0), (0, 2, 0)])

    # ID=1 alive t=0,1 — dies at t=2; ID=2 appears at t=2 in same slot
    ids = _ids(B, T, N,
               [(0, 0, 0, 1), (0, 1, 0, 1),   # ID=1
                (0, 2, 0, 2)])                   # ID=2 (slot 0 reused)

    loss, info = compute_loss(pred_s, pred_l, gt, mask, state_ids=ids)

    assert torch.isfinite(loss), "Loss is non-finite after target death"
    assert info["n_matched"] == 3, (
        f"Expected 3 matched pairs (1 per frame), got {info['n_matched']}"
    )


# ── Test 4: All targets absent ─────────────────────────────────────────────────

def test_all_absent_finite():
    """
    No live targets in any frame.  Loss must be finite (exist BCE on zeros)
    and gradients must flow to pred_logits.
    """
    B, T, N = 2, 4, N_MAX

    pred_s = torch.zeros(B, T, N, D_STATE, requires_grad=True)
    pred_l = torch.zeros(B, T, N, requires_grad=True)
    gt     = torch.zeros(B, T, N, D_STATE)
    mask   = torch.zeros(B, T, N, dtype=torch.bool)
    ids    = torch.full((B, T, N), -1, dtype=torch.int32)

    loss, info = compute_loss(pred_s, pred_l, gt, mask, state_ids=ids)

    assert torch.isfinite(loss),               "Loss must be finite when all targets absent"
    loss.backward()
    assert pred_l.grad is not None,            "pred_logits must receive gradient"
    assert torch.all(torch.isfinite(pred_l.grad)), "pred_logits gradient must be finite"
    assert info["n_matched"] == 0
    assert info["n_gt"]      == 0


# ── Test 5: GT slot reordering is transparent ─────────────────────────────────

def test_gt_slot_reorder_stable():
    """
    Two targets (ID=10, ID=20) exist for all T=2 frames.
    Case A: both frames keep the same slot assignment (10→slot0, 20→slot1).
    Case B: at frame 1 the simulator packs them differently (10→slot1, 20→slot0).

    Because the matcher follows IDs, not slots, the total loss must be identical.
    Predictions fixed: query 0 at +1 (tracks ID=10), query 1 at −1 (tracks ID=20).
    """
    B, T, N = 1, 2, N_MAX

    # GT — values match the query that will track each ID
    gt_a = torch.zeros(B, T, N, D_STATE)
    gt_a[0, 0, 0, 0] =  1.0   # frame 0: ID=10 → slot 0
    gt_a[0, 0, 1, 0] = -1.0   # frame 0: ID=20 → slot 1
    gt_a[0, 1, 0, 0] =  1.0   # frame 1: ID=10 → slot 0
    gt_a[0, 1, 1, 0] = -1.0   # frame 1: ID=20 → slot 1

    # Case B: frame 1 swaps the slots (IDs follow)
    gt_b = gt_a.clone()
    gt_b[0, 1, 0, 0] = -1.0   # frame 1: ID=20 now in slot 0
    gt_b[0, 1, 1, 0] =  1.0   # frame 1: ID=10 now in slot 1

    # Fixed predictions: query 0 → +1, query 1 → −1
    pred_s = torch.zeros(B, T, N, D_STATE)
    pred_s[0, :, 0, 0] =  1.0
    pred_s[0, :, 1, 0] = -1.0
    pred_s = pred_s.detach().requires_grad_(True)
    pred_l = torch.zeros(B, T, N, requires_grad=True)

    mask = _mask(B, T, N, [(0, t, n) for t in range(T) for n in range(2)])

    ids_a = _ids(B, T, N,
                 [(0, 0, 0, 10), (0, 0, 1, 20),
                  (0, 1, 0, 10), (0, 1, 1, 20)])

    ids_b = _ids(B, T, N,
                 [(0, 0, 0, 10), (0, 0, 1, 20),   # frame 0 same
                  (0, 1, 0, 20), (0, 1, 1, 10)])   # frame 1 swapped

    loss_a, _ = compute_loss(pred_s, pred_l, gt_a, mask, state_ids=ids_a)
    loss_b, _ = compute_loss(pred_s, pred_l, gt_b, mask, state_ids=ids_b)

    assert abs(loss_a.item() - loss_b.item()) < 1e-5, (
        f"Slot reorder changed loss: {loss_a:.6f} vs {loss_b:.6f}"
    )


# ── Test 6: use_context guard ─────────────────────────────────────────────────

def test_context_guard_raises():
    """
    AgnosticTransformer with use_context=True must raise ValueError when
    contexts=None is passed to forward().
    """
    model = AgnosticTransformer(
        N_MAX=3, d_model=16, n_heads=2, n_enc_layers=1,
        n_dec_layers=1, dim_ff=32, dropout=0.0,
        use_quality=False, use_context=True,
    )
    B, T, M = 1, 4, 10
    dets  = torch.zeros(B, T, M, 8)
    valid = torch.ones(B, T, M, dtype=torch.bool)

    with pytest.raises(ValueError, match="use_context=True"):
        model(dets, valid, contexts=None)
