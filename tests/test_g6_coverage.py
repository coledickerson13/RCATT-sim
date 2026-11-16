"""
test_g6_coverage.py – G6: Detection-probability and measurement-uncertainty
            curves versus range for C1–C8 under all SIC states
===========================================================================
Validates the system-level sensing coverage implied by the physics model:

    P_D(R) = exp(−η / (1 + γ_mean(R)))
    γ_mean(R) = n_pil · compute_sinr(R, az, el, σ_mean, beam, cfg, sic_db)
    σ_meas(R) = √(diag(compute_r_sim(compute_crlb_variances(·))))

Beam 12 boresight (az=7.5°, el=21.25°) is used throughout; it maximises
SINR within the grid and gives the most conservative (tightest) coverage
for a fixed-beam scenario, bounding the actual scanned-beam performance.

Pass criteria
-------------
G6-1  P_D(R) is strictly decreasing with R for every config × SIC combo.
G6-2  P_D is strictly increasing with SIC isolation at every range.
G6-3  For every train config (C1–C4, C6, C8) under SIC=120 dB,
      P_D(R=300 m) ≥ 0.50 (coverage reaches the surveillance boundary).
G6-4  σ_R(R) is non-decreasing with R for all configs (noise grows with range).
G6-5  σ_R, σ_vr approach their floors at minimum valid range (R=120 m, high SNR).
G6-6  At R=120 m (SIC=120 dB), all configs satisfy
      σ_R ≤ 2 m, σ_vr ≤ 2 m/s, σ_az ≤ 2°, σ_el ≤ 2°.
G6-7  Better resource config → higher P_D at R=150 m under SIC=110 dB
      (C8 ≥ C4, C6 ≥ C2 in terms of SINR; checked for ordering).
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
    DETECTION_THRESHOLD, RESOURCE_CONFIGS, SIC_STATES_DB, SIGMA_RCS_MEAN,
    FLOOR_RANGE_M, FLOOR_VEL_MPS, FLOOR_AZ_DEG, FLOOR_EL_DEG,
    SURV_RHO_MAX, GNB_POS,
    SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
    SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
    UAV_ALT_MIN, UAV_ALT_MAX,
    N_BEAMS,
)
from simulator.measurement_model import compute_sinr, compute_crlb_variances, compute_r_sim
from simulator.beam_steering import BEAM_CENTRES as _BEAM_CENTRES_RAW, beam_gain_tx


# ---------------------------------------------------------------------------
# Shared geometry – beam 12 boresight, maximises SINR
# ---------------------------------------------------------------------------
_AZ_B12, _EL_B12, _BEAM = 7.5, 21.25, 12
_R_GRID = np.array([120., 150., 200., 250., 300.])  # metres — all physically valid (UAV_ALT_MIN=50 m, gNB=10 m → min slant ≈110 m)
_ALL_CFGS   = list(RESOURCE_CONFIGS.keys())
_TRAIN_CFGS = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'train']
_FLOORS = np.array([FLOOR_RANGE_M, FLOOR_VEL_MPS, FLOOR_AZ_DEG, FLOOR_EL_DEG])


def _pd(R, cfg_key, sic_db):
    """Swerling-I P_D at slant range R, config cfg_key, SIC sic_db."""
    cfg = RESOURCE_CONFIGS[cfg_key]
    xi  = compute_sinr(R, _AZ_B12, _EL_B12, SIGMA_RCS_MEAN, _BEAM, cfg, sic_db)
    snr = cfg['n_pil'] * xi
    return float(np.exp(-DETECTION_THRESHOLD / (1.0 + snr)))


def _sigma_meas(R, cfg_key, sic_db):
    """σ_meas = sqrt(diag(R_sim)), shape (4,)."""
    cfg  = RESOURCE_CONFIGS[cfg_key]
    xi   = compute_sinr(R, _AZ_B12, _EL_B12, SIGMA_RCS_MEAN, _BEAM, cfg, sic_db)
    crlb = compute_crlb_variances(xi, cfg, _AZ_B12, _EL_B12)
    return np.sqrt(np.diag(compute_r_sim(crlb)))


# ---------------------------------------------------------------------------
# G6-1: P_D strictly decreasing with range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
@pytest.mark.parametrize("sic_db", SIC_STATES_DB)
def test_pd_decreasing_with_range(cfg_key, sic_db):
    """P_D(R) is strictly decreasing with slant range."""
    pd_vals = np.array([_pd(R, cfg_key, sic_db) for R in _R_GRID])
    diffs   = np.diff(pd_vals)
    assert np.all(diffs < 0), (
        f"{cfg_key} SIC={sic_db}: P_D not strictly decreasing. "
        f"Ranges={_R_GRID[np.where(diffs >= 0)[0]]}, P_D={pd_vals}"
    )


# ---------------------------------------------------------------------------
# G6-2: P_D increases with SIC isolation at every range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
def test_pd_increases_with_sic_at_all_ranges(cfg_key):
    """P_D is strictly increasing with SIC isolation for all ranges in _R_GRID."""
    sic_sorted = sorted(SIC_STATES_DB)
    for R in _R_GRID:
        pd_vals = [_pd(R, cfg_key, sic) for sic in sic_sorted]
        diffs   = np.diff(pd_vals)
        assert np.all(diffs > 0), (
            f"{cfg_key} R={R}: P_D not increasing with SIC. "
            f"P_D={pd_vals} at SIC={sic_sorted}"
        )


# ---------------------------------------------------------------------------
# G6-3: P_D ≥ 50% at the 300 m surveillance boundary for all train configs
# ---------------------------------------------------------------------------
#
# Previous formulation (_r50 ≥ 80 m) was vacuously true: _r50() searches
# 110–400 m and returns r_max=400 when P_D > 0.5 everywhere, so the bound
# r50 ≥ 80 was guaranteed by the search interval itself regardless of model
# performance.  The correct criterion is P_D ≥ 50% at the actual 300 m
# surveillance boundary.  _r50 is kept as an informational helper.

def _r50(cfg_key, sic_db, r_min=110.0, r_max=400.0, n_pts=200):
    """Numerically find the range where P_D crosses 0.5.

    Returns r_max when P_D > 0.5 throughout (coverage extends beyond the
    search interval); callers should treat the return value accordingly.
    """
    rs  = np.linspace(r_min, r_max, n_pts)
    pds = np.array([_pd(R, cfg_key, sic_db) for R in rs])
    idx = np.where(pds < 0.5)[0]
    if len(idx) == 0:
        return r_max   # P_D always > 0.5 within search interval
    return float(rs[idx[0]])


@pytest.mark.parametrize("cfg_key", _TRAIN_CFGS)
def test_pd_at_coverage_boundary_train_configs(cfg_key):
    """Train configs under SIC=120 dB achieve P_D ≥ 0.5 at R=300 m (coverage boundary).

    300 m is the outer edge of the 3-D surveillance volume (RHO_MAX).  Asserting
    P_D ≥ 0.5 there is the meaningful coverage criterion — if the 50%-crossover
    range is beyond 400 m (search limit), it is reported as '>400 m'.
    """
    pd_300 = _pd(300.0, cfg_key, 120.0)
    assert pd_300 >= 0.50, (
        f"{cfg_key}: P_D({pd_300:.4f}) < 0.50 at R=300 m, SIC=120 dB"
    )
    r50 = _r50(cfg_key, 120.0)
    r50_str = f">400 m" if r50 >= 400.0 else f"{r50:.1f} m"
    # R_50 must also reach at least the coverage boundary
    assert r50 >= 300.0, (
        f"{cfg_key}: R_50={r50_str} < 300 m at SIC=120 dB — coverage too short"
    )


# ---------------------------------------------------------------------------
# G6-4: σ_R non-decreasing with range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
def test_sigma_R_nondecreasing_with_range(cfg_key):
    """σ_R (range std) is non-decreasing with slant range (SIC=120 dB)."""
    sigmas = np.array([_sigma_meas(R, cfg_key, 120.0)[0] for R in _R_GRID])
    diffs  = np.diff(sigmas)
    assert np.all(diffs >= -1e-12), (
        f"{cfg_key}: σ_R not non-decreasing. sigmas={sigmas}"
    )


# ---------------------------------------------------------------------------
# G6-5: Measurement floors are approached at short range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _TRAIN_CFGS)
def test_sigma_approaches_floor_at_short_range(cfg_key):
    """At R=120 m (minimum valid range) SIC=120 dB, σ_meas ≤ 2.5× floor.

    120 m is the nearest physically valid slant range (UAV_ALT_MIN=50 m,
    gNB altitude=10 m → min range at el=21.25° ≈ 110 m).  High SNR at
    this range drives σ_meas close to the noise floor.
    """
    sigma = _sigma_meas(120.0, cfg_key, 120.0)
    for i, (s, f) in enumerate(zip(sigma, _FLOORS)):
        assert s <= 2.5 * f, (
            f"{cfg_key} dim={i}: σ={s:.4g} m(°), floor={f:.4g} → ratio={s/f:.2f} > 2.5"
        )


# ---------------------------------------------------------------------------
# G6-6: Measurement quality at R=120 m SIC=120 dB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", _ALL_CFGS)
def test_sigma_at_120m_within_2units(cfg_key):
    """At R=120 m SIC=120 dB, σ_R ≤ 2 m, σ_vr ≤ 2 m/s, σ_az ≤ 2°, σ_el ≤ 2°.

    120 m is the minimum physically valid slant range (UAV_ALT_MIN=50 m above ground,
    gNB at 10 m altitude).  At this range all configs have high enough SINR that
    σ_meas stays well within 2 × (noise floor).
    """
    sigma = _sigma_meas(120.0, cfg_key, 120.0)
    limits = [2.0, 2.0, 2.0, 2.0]
    labels = ['R (m)', 'vr (m/s)', 'az (°)', 'el (°)']
    for s, lim, lbl in zip(sigma, limits, labels):
        assert s <= lim, (
            f"{cfg_key} σ_{lbl}={s:.4f} > {lim} at R=120 m, SIC=120 dB"
        )


# ---------------------------------------------------------------------------
# G6-8: Spatial detection coverage — best-beam P_D across surveillance volume
# ---------------------------------------------------------------------------
#
# Samples UAV positions uniformly throughout the 3-D surveillance volume via
# the same rejection-sampling geometry used by _birth_state / _gen_clutter.
# For each position the best beam (highest gain) is selected and P_D is
# computed at SIC=120 dB.  This catches configurations where good beam-center
# performance masks poor off-boresight coverage.

def _sample_surv_positions(n: int, seed: int = 6800):
    """Return n (az_deg, el_deg, R_m) tuples sampled uniformly inside the
    surveillance volume by rejection sampling, mirroring _birth_state."""
    rng   = np.random.default_rng(seed)
    x_half = SURV_RHO_MAX * np.sin(np.deg2rad(SURV_AZ_MAX_DEG))
    pts   = []
    while len(pts) < n:
        x = rng.uniform(-x_half, x_half)
        y = rng.uniform(0.0, SURV_RHO_MAX)
        z = rng.uniform(UAV_ALT_MIN, UAV_ALT_MAX)
        dx, dy, dz = x - GNB_POS[0], y - GNB_POS[1], z - GNB_POS[2]
        R  = float(np.sqrt(dx*dx + dy*dy + dz*dz))
        if R > SURV_RHO_MAX or R < 1e-6:
            continue
        az = float(np.degrees(np.arctan2(dx, dy)))
        el = float(np.degrees(np.arcsin(np.clip(dz / R, -1.0, 1.0))))
        if not (SURV_AZ_MIN_DEG <= az <= SURV_AZ_MAX_DEG):
            continue
        if not (SURV_EL_MIN_DEG <= el <= SURV_EL_MAX_DEG):
            continue
        pts.append((az, el, R))
    return pts


def _pd_best_beam(az, el, R, cfg_key, sic_db):
    """P_D at (az, el, R) using the highest-gain beam for this direction."""
    gains    = np.array([beam_gain_tx(az, el, _BEAM_CENTRES_RAW, b)
                         for b in range(N_BEAMS)])
    best_b   = int(np.argmax(gains))
    cfg      = RESOURCE_CONFIGS[cfg_key]
    xi       = compute_sinr(R, az, el, SIGMA_RCS_MEAN, best_b, cfg, sic_db)
    snr      = cfg['n_pil'] * xi
    return float(np.exp(-DETECTION_THRESHOLD / (1.0 + snr)))


@pytest.mark.parametrize("cfg_key", _TRAIN_CFGS)
def test_spatial_pd_coverage(cfg_key):
    """P_D ≥ 0.5 for ≥ 90% of positions sampled across the surveillance volume
    at SIC=120 dB using best-beam selection (1000-point Monte Carlo).

    This complements the beam-12 boresight curve (G6-1 to G6-7) with an
    off-boresight spatial average that is representative of training-data
    conditions.
    """
    N_PTS = 1_000
    pts   = _sample_surv_positions(N_PTS, seed=6801)
    n_pass = sum(1 for (az, el, R) in pts if _pd_best_beam(az, el, R, cfg_key, 120.0) >= 0.50)
    frac = n_pass / N_PTS
    assert frac >= 0.90, (
        f"{cfg_key}: only {frac*100:.1f}% of spatial samples achieve P_D ≥ 0.5 "
        f"at SIC=120 dB — expected ≥ 90%"
    )


# ---------------------------------------------------------------------------
# G6-7: Resource ordering – more pilots/bandwidth → higher P_D
# ---------------------------------------------------------------------------

def test_resource_ordering_pd_at_150m():
    """Higher n_pil config has higher P_D at R=150 m SIC=110 dB.

    C8 (n_pil=50688) ≥ C4 (n_pil=39936) ≥ C3 (n_pil=18432) ≥ C1 (n_pil=9216).
    These share the same n_slot=16 group so the comparison is clean.
    """
    R, sic = 150.0, 110.0
    cfg_order = ['C1', 'C3', 'C4', 'C8']   # increasing n_pil for n_slot=16 group
    pd_vals = [_pd(R, k, sic) for k in cfg_order]
    for i in range(len(pd_vals) - 1):
        assert pd_vals[i] < pd_vals[i + 1], (
            f"P_D ordering failed: {cfg_order[i]}({pd_vals[i]:.4f}) ≥ "
            f"{cfg_order[i+1]}({pd_vals[i+1]:.4f}) at R={R} m, SIC={sic} dB"
        )
