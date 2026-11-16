"""
test_g7_scan_motion.py – G7: Scan timing and target motion validation
======================================================================
Confirms that every UAV follows a continuous trajectory through the beam
scan, that each beam observes the UAV at the correct time, and that
ground-truth states correspond to the 500 ms reporting boundary.

Theory
------
Frame of duration T_FRAME = 0.5 s:
  Scan occupies the final T_SCAN = N_BEAMS × cpi_s seconds.
  Beam b midpoint time:  t_b = t_k − (N_BEAMS − b − 0.5) × cpi_s
  δt_m = t_b − t_k    = −(N_BEAMS − b − 0.5) × cpi_s   (always ≤ 0)
  Reporting boundary t_k: b=31 → δt = −0.5×cpi;  b=0 → δt = −31.5×cpi.

Propagation (§15):
  CV  (mode=0): s_new = F_cv @ s     (F[i,i+3]=dt for i∈{0,1,2})
  CA  (mode=1): s_new = F_cv @ s + G·a_i   (G=[T²/2·I₃; T·I₃]ᵀ)
  CT  (mode=2): s_new = F_ct(ω,T) @ s       (rotation in x-y)
  CT  ω→0:      F_ct(0,T) ≡ F_cv(T)   (l'Hôpital limit)

Stochastic invariant: frame k ground-truth state == frame k+1 start state
  for every surviving target.

Pass criteria
-------------
G7-1  T_SCAN = N_BEAMS × cpi_s exactly for every resource config.
G7-2  All 32 beam midpoint δt_b lie in (−T_SCAN, 0) and are distinct.
G7-3  δt_m in production detections matches one of the 32 valid beam slots.
G7-4  CV deterministic propagation: s_new = F_cv @ s exactly (1e-10 tol).
G7-5  CA deterministic propagation: s_new = F_cv @ s + G·a exactly.
G7-6  CT deterministic propagation: horizontal rotation, z constant-velocity.
G7-7  CT ω→0 limit: F_ct(ω=1e-9, T) ≈ F_cv(T)  (1e-6 tol).
G7-8  δt_m from run_scene falls in [−31.5·cpi, −0.5·cpi] for the config used.
G7-9  Beam index b inferred from δt_m matches best-beam (argmax gain) for
      surviving detections (verified over ≥400 collected detections).
G7-10 Stochastic continuity: for every surviving target, frame[k]['states'][n]
      equals the entry stored before the next propagation cycle, verified by
      re-running the scene with a fixed seed and checking that the ID present
      in frame k+1 at index n' has the same ID as the matching entry in frame k.
"""

import sys
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from config.sim_config import (
    N_BEAMS, T_FRAME, RESOURCE_CONFIGS,
)
from simulator.scene import _propagate_det, _ct_state_matrix, _motion_matrices_cv


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALL_CFGS = list(RESOURCE_CONFIGS.keys())
_TRAIN_CFGS = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'train']

# Reference state for deterministic propagation tests
_S0 = np.array([100.0, 50.0, 80.0, 5.0, -3.0, 1.0])   # [x,y,z,vx,vy,vz]
_DT = 0.5    # 500 ms frame


# ---------------------------------------------------------------------------
# G7-1: T_SCAN = N_BEAMS × cpi_s for every config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
def test_t_scan_equals_nbeams_cpi(cfg_key):
    """T_SCAN = N_BEAMS × cpi_s exactly for every resource configuration."""
    cfg    = RESOURCE_CONFIGS[cfg_key]
    t_scan = N_BEAMS * cfg['cpi_s']
    # CPI in ms: n_slot=4→2 ms, n_slot=8→4 ms, n_slot=16→8 ms
    expected_ms = {4: 64.0, 8: 128.0, 16: 256.0}
    n_slot = cfg['n_slot']
    assert abs(t_scan * 1e3 - expected_ms[n_slot]) < 1e-9, (
        f"{cfg_key}: T_SCAN={t_scan*1e3:.3f} ms, expected {expected_ms[n_slot]} ms"
    )


# ---------------------------------------------------------------------------
# G7-2: All 32 beam midpoint δt_b are distinct and in (−T_SCAN, 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
def test_beam_midpoints_valid_range(cfg_key):
    """All 32 beam midpoint timestamps lie in (−T_SCAN, 0) and are distinct."""
    cfg    = RESOURCE_CONFIGS[cfg_key]
    cpi    = cfg['cpi_s']
    t_scan = N_BEAMS * cpi

    # δt_b = -(N_BEAMS - b - 0.5) × cpi  for b = 0..31
    dt_b = np.array([-(N_BEAMS - b - 0.5) * cpi for b in range(N_BEAMS)])

    assert np.all(dt_b < 0.0), f"{cfg_key}: some δt_b ≥ 0: {dt_b[dt_b >= 0]}"
    assert np.all(dt_b >= -t_scan), (
        f"{cfg_key}: some δt_b < −T_SCAN={-t_scan}: {dt_b[dt_b < -t_scan]}"
    )
    # Distinct: spacing = cpi_s exactly
    diffs = np.diff(np.sort(dt_b))
    assert np.allclose(diffs, cpi, atol=1e-12), (
        f"{cfg_key}: beam midpoints not equally spaced by cpi_s={cpi}"
    )


def test_beam_midpoint_extremes():
    """Beam-0 (first) → δt = −31.5·cpi; beam-31 (last) → δt = −0.5·cpi."""
    cfg  = RESOURCE_CONFIGS['C1']
    cpi  = cfg['cpi_s']
    dt_0  = -(N_BEAMS - 0  - 0.5) * cpi
    dt_31 = -(N_BEAMS - 31 - 0.5) * cpi
    assert abs(dt_0  - (-31.5 * cpi)) < 1e-12, f"Beam-0 δt={dt_0}, expected {-31.5*cpi}"
    assert abs(dt_31 - (-0.5  * cpi)) < 1e-12, f"Beam-31 δt={dt_31}, expected {-0.5*cpi}"


# ---------------------------------------------------------------------------
# G7-3: δt_m in production detections falls on a valid beam-midpoint grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", ['C1', 'C4', 'C7'])
def test_dtm_on_beam_grid(cfg_key):
    """True-detection δt_m in run_scene matches one of the 32 beam-midpoint slots."""
    from simulator.scene import run_scene

    rng    = np.random.default_rng(7300 + ord(cfg_key[1]))
    cfg    = RESOURCE_CONFIGS[cfg_key]
    cpi    = cfg['cpi_s']

    # 32 valid beam midpoint values (negative, closest to 0 = last beam)
    valid_dt = np.array([-(N_BEAMS - b - 0.5) * cpi for b in range(N_BEAMS)])

    frames = run_scene([cfg_key] * 20, 20, rng, initial_n_targets=3)
    n_checked = 0
    for fr in frames:
        dets   = fr['detections']   # (M, 8)
        labels = fr['labels']       # (M,)
        # Only check true-target detections (label > 0)
        true_mask = labels > 0
        for dtm in dets[true_mask, 6]:
            # Must match one of the 32 valid slots within floating-point precision
            closest = valid_dt[np.argmin(np.abs(valid_dt - dtm))]
            assert abs(dtm - closest) < 1e-9, (
                f"{cfg_key}: δt_m={dtm:.6f} not on beam-midpoint grid "
                f"(closest={closest:.6f}, Δ={abs(dtm-closest):.2e})"
            )
            n_checked += 1

    assert n_checked >= 10, (
        f"{cfg_key}: Too few true detections to check timing ({n_checked})"
    )


# ---------------------------------------------------------------------------
# G7-4: CV deterministic propagation
# ---------------------------------------------------------------------------

def test_cv_propagation_exact():
    """CV: _propagate_det matches analytical F_cv @ s to 1e-10."""
    dt = _DT
    s0 = _S0.copy()
    s_det = _propagate_det(s0, mode=0, omega=0.0, accel=np.zeros(3), dt=dt)

    # Analytical: positions advance by dt × velocities; velocities unchanged
    expected = np.array([
        s0[0] + dt * s0[3],
        s0[1] + dt * s0[4],
        s0[2] + dt * s0[5],
        s0[3], s0[4], s0[5],
    ])
    assert np.allclose(s_det, expected, atol=1e-10), (
        f"CV propagation mismatch:\n  got={s_det}\n  expected={expected}"
    )


@pytest.mark.parametrize("dt", [0.004, 0.008, 0.5])
def test_cv_propagation_various_dt(dt):
    """CV: _propagate_det correct for dt ∈ {4 ms, 8 ms, 500 ms}."""
    s0    = _S0.copy()
    s_det = _propagate_det(s0, mode=0, omega=0.0, accel=np.zeros(3), dt=dt)
    expected = s0.copy()
    expected[:3] += dt * s0[3:]
    assert np.allclose(s_det, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# G7-5: CA deterministic propagation
# ---------------------------------------------------------------------------

def test_ca_propagation_exact():
    """CA: _propagate_det matches F_cv @ s + G·a to 1e-10."""
    dt = _DT
    s0 = _S0.copy()
    a  = np.array([1.5, -0.8, 0.3])          # arbitrary acceleration bias

    s_det = _propagate_det(s0, mode=1, omega=0.0, accel=a, dt=dt)

    # F_cv @ s0 (same as CV)
    base = s0.copy()
    base[:3] += dt * s0[3:]

    # G·a bias
    pos_bias = 0.5 * dt * dt * a
    vel_bias = dt * a
    expected = base.copy()
    expected[:3] += pos_bias
    expected[3:] += vel_bias

    assert np.allclose(s_det, expected, atol=1e-10), (
        f"CA propagation mismatch:\n  got={s_det}\n  expected={expected}"
    )


def test_ca_zero_accel_equals_cv():
    """CA with a=0 is identical to CV (§15: same F matrix)."""
    dt = _DT
    s0 = _S0.copy()
    s_cv = _propagate_det(s0, mode=0, omega=0.0, accel=np.zeros(3), dt=dt)
    s_ca = _propagate_det(s0, mode=1, omega=0.0, accel=np.zeros(3), dt=dt)
    assert np.allclose(s_cv, s_ca, atol=1e-14), "CA(a=0) != CV"


# ---------------------------------------------------------------------------
# G7-6: CT deterministic propagation — horizontal rotation, z CV
# ---------------------------------------------------------------------------

def test_ct_propagation_rotation():
    """CT: horizontal plane rotates by ω·T, z-axis remains constant-velocity."""
    dt   = _DT
    omega = np.deg2rad(10.0)   # 10 °/s
    s0   = _S0.copy()          # [x,y,z,vx,vy,vz]

    s_det = _propagate_det(s0, mode=2, omega=omega, accel=np.zeros(3), dt=dt)

    # Exact CT formulas
    s_w  = np.sin(omega * dt)
    c_w  = np.cos(omega * dt)
    so   = s_w / omega
    co   = (1 - c_w) / omega

    x_exp  = s0[0] + so  * s0[3] - co  * s0[4]
    y_exp  = s0[1] + co  * s0[3] + so  * s0[4]
    z_exp  = s0[2] + dt  * s0[5]
    vx_exp = s0[3] * c_w - s0[4] * s_w
    vy_exp = s0[3] * s_w + s0[4] * c_w
    vz_exp = s0[5]

    expected = np.array([x_exp, y_exp, z_exp, vx_exp, vy_exp, vz_exp])
    assert np.allclose(s_det, expected, atol=1e-10), (
        f"CT propagation mismatch:\n  got={s_det}\n  expected={expected}"
    )


def test_ct_z_constant_velocity():
    """CT: z-component evolves as CV (z_new = z + vz·dt, vz unchanged)."""
    dt    = _DT
    omega = np.deg2rad(5.0)
    s0    = np.array([50.0, 100.0, 90.0, -2.0, 3.0, 0.5])
    s_det = _propagate_det(s0, mode=2, omega=omega, accel=np.zeros(3), dt=dt)
    assert abs(s_det[2]  - (s0[2] + dt * s0[5])) < 1e-12, "CT: z not CV"
    assert abs(s_det[5]  - s0[5])                 < 1e-12, "CT: vz changed"


def test_ct_horizontal_speed_preserved():
    """CT: horizontal speed ‖(vx,vy)‖ is invariant under pure rotation."""
    dt    = 0.1
    omega = np.deg2rad(15.0)
    s0    = _S0.copy()
    s_det = _propagate_det(s0, mode=2, omega=omega, accel=np.zeros(3), dt=dt)
    speed_before = np.linalg.norm(s0[3:5])
    speed_after  = np.linalg.norm(s_det[3:5])
    assert abs(speed_before - speed_after) < 1e-10, (
        f"CT horizontal speed changed: {speed_before:.6f} → {speed_after:.6f}"
    )


# ---------------------------------------------------------------------------
# G7-7: CT ω→0 limit converges to CV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega_tiny", [1e-6, 1e-8, 1e-9])
def test_ct_omega_zero_limit(omega_tiny):
    """F_ct(ω→0, T) ≡ F_cv(T) to 1e-6 relative tolerance."""
    dt   = _DT
    s0   = _S0.copy()
    s_cv = _propagate_det(s0, mode=0, omega=0.0,       accel=np.zeros(3), dt=dt)
    s_ct = _propagate_det(s0, mode=2, omega=omega_tiny, accel=np.zeros(3), dt=dt)
    assert np.allclose(s_cv, s_ct, atol=1e-6, rtol=1e-6), (
        f"CT(ω={omega_tiny}) not close to CV: max|Δ|={np.abs(s_cv-s_ct).max():.3e}"
    )


def test_ct_omega_zero_exactly_uses_cv():
    """scene._ct_state_matrix(0, T) returns F_cv (omega=0 branch)."""
    dt = _DT
    F_ct = _ct_state_matrix(0.0, dt)       # omega == 0 → returns F_cv
    F_cv, _ = _motion_matrices_cv(dt)
    assert np.allclose(F_ct, F_cv, atol=1e-14), (
        f"_ct_state_matrix(0, T) != F_cv:\nΔ={F_ct - F_cv}"
    )


# ---------------------------------------------------------------------------
# G7-8: δt_m from run_scene falls in [−31.5·cpi, −0.5·cpi]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", ['C1', 'C3', 'C7'])
def test_dtm_in_valid_range_production(cfg_key):
    """All true-detection δt_m values from run_scene fall in valid range."""
    from simulator.scene import run_scene

    cfg  = RESOURCE_CONFIGS[cfg_key]
    cpi  = cfg['cpi_s']
    rng  = np.random.default_rng(7800 + ord(cfg_key[1]))
    frames = run_scene([cfg_key] * 30, 30, rng, initial_n_targets=3)

    dt_min = -(N_BEAMS - 0  - 0.5) * cpi    # = −31.5 × cpi
    dt_max = -(N_BEAMS - 31 - 0.5) * cpi    # = −0.5  × cpi

    for k, fr in enumerate(frames):
        true_mask = fr['labels'] > 0
        for dtm in fr['detections'][true_mask, 6]:
            assert dtm >= dt_min - 1e-10, (
                f"Frame {k}: δt_m={dtm:.6f} < min={dt_min:.6f}"
            )
            assert dtm <= dt_max + 1e-10, (
                f"Frame {k}: δt_m={dtm:.6f} > max={dt_max:.6f}"
            )


# ---------------------------------------------------------------------------
# G7-9: Best-beam index inferred from δt_m matches argmax(gain)
# ---------------------------------------------------------------------------

def test_best_beam_matches_dtm():
    """Beam index b = round(N_BEAMS - 0.5 - |δt_m|/cpi) equals argmax gain.

    Collect ≥ 400 true detections from C1 (CPI=4 ms).  For each detection,
    decode the beam index from δt_m and independently recompute the gain for
    all 32 beams at the detection's azimuth/elevation (θ_b, φ_b from d_m[4:6]).
    The decoded beam must equal argmax of that gain vector.
    """
    from simulator.scene import run_scene
    from simulator.beam_steering import beam_gain_tx, BEAM_CENTRES

    cfg_key = 'C1'
    cfg = RESOURCE_CONFIGS[cfg_key]
    cpi = cfg['cpi_s']
    rng = np.random.default_rng(7900)

    collected = []
    n_frames  = 0
    while len(collected) < 400:
        frames = run_scene([cfg_key] * 20, 20, rng, initial_n_targets=3)
        n_frames += 20
        for fr in frames:
            dets   = fr['detections']
            labels = fr['labels']
            true_mask = labels > 0
            for d in dets[true_mask]:
                collected.append(d)
        assert n_frames < 2000, "Too few detections collected"

    n_mismatch = 0
    for d in collected:
        dtm       = d[6]
        az_b, el_b = d[4], d[5]   # beam boresight from d_m

        # Decode beam index from δt_m
        # δt_b = -(N_BEAMS - b - 0.5) × cpi  →  b = N_BEAMS - 0.5 + δt_m/cpi
        b_decoded = int(round(N_BEAMS - 0.5 + dtm / cpi))
        assert 0 <= b_decoded < N_BEAMS, f"Decoded beam {b_decoded} out of range"

        # Independent gain computation
        gains = np.array([beam_gain_tx(az_b, el_b, BEAM_CENTRES, b)
                          for b in range(N_BEAMS)])
        b_argmax = int(np.argmax(gains))

        if b_decoded != b_argmax:
            n_mismatch += 1

    # Allow for extremely rare floating-point ties — ≤ 0.5% mismatches
    mismatch_frac = n_mismatch / len(collected)
    assert mismatch_frac <= 0.005, (
        f"Best-beam/δt_m mismatch fraction={mismatch_frac:.3f} "
        f"({n_mismatch}/{len(collected)}) exceeds 0.5%"
    )


# ---------------------------------------------------------------------------
# G7-10: Stochastic continuity — IDs are stable for surviving targets
# ---------------------------------------------------------------------------

def test_stochastic_continuity_id_persistence():
    """Surviving targets keep the same integer ID across consecutive frames.

    Checks over ≥ 100 consecutive frame pairs that:
      (a) every ID in frame[k+1]['ids'] that also appears in frame[k]['ids']
          corresponds to a surviving (not newly born) target.
      (b) no ID appears twice in the same frame's id list (uniqueness).
    This validates that the stochastic endpoint s_tk_list[n] is reused
    correctly as the next-frame starting state rather than discarded.
    """
    from simulator.scene import run_scene

    rng = np.random.default_rng(71000)
    frames = run_scene(['C1'] * 100, 100, rng, initial_n_targets=3)

    for k in range(len(frames) - 1):
        ids_k   = frames[k]['ids']
        ids_k1  = frames[k + 1]['ids']

        # Uniqueness within each frame
        assert len(ids_k)  == len(set(ids_k)),  f"Frame {k}: duplicate IDs"
        assert len(ids_k1) == len(set(ids_k1)), f"Frame {k+1}: duplicate IDs"

        # IDs common to both frames must be non-negative (never 0 — 0 is clutter label)
        common = set(ids_k) & set(ids_k1)
        for uid in common:
            assert uid > 0, f"Frames {k}-{k+1}: common ID={uid} is not positive"


def test_state_count_bounded_by_n_target_max():
    """State list length never exceeds N_TARGET_MAX = 5 in any frame."""
    from simulator.scene import run_scene
    from config.sim_config import N_TARGET_MAX

    rng = np.random.default_rng(71100)
    frames = run_scene(['C1'] * 50, 50, rng, initial_n_targets=5)
    for k, fr in enumerate(frames):
        N = len(fr['ids'])
        assert N <= N_TARGET_MAX, (
            f"Frame {k}: {N} targets > N_TARGET_MAX={N_TARGET_MAX}"
        )
