"""
test_g5_detector.py – G5: Fixed-threshold detector validation
=============================================================
Validates the Ricean matched-filter / Swerling-I detection model:

    z = |√γ + noise|²,  noise ~ CN(0,1)
    detect iff  z > η = DETECTION_THRESHOLD = −ln(P_FA)

Theory
------
H0 (no target, γ=0):  z ~ Exp(1)  →  P_FA = exp(−η)
Swerling-I (γ ~ Exp(γ_mean)):  P_D = exp(−η / (1 + γ_mean))

Pass criteria
-------------
G5-1  DETECTION_THRESHOLD = −ln(P_FA) exactly (1e-12 tolerance).
G5-2  Null z mean = 1 and variance = 1 (N=20 k, 3σ MC tolerance).
G5-3  Swerling-I P_D formula matches Monte Carlo within 2.5% at
      SNR_mean ∈ {5, 20, 100} (N=50 k per point, 3σ_MC ≪ 2.5%).
G5-4  P_D is monotone in SNR_mean across [0.1 … 100].
G5-5  P_D is monotone in SIC isolation for C1 at R=200 m.
"""

import sys
import os

# Add rcatt_sim root to path so production imports work
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from config.sim_config import (
    DETECTION_THRESHOLD, P_FA, RESOURCE_CONFIGS, SIC_STATES_DB, SIGMA_RCS_MEAN,
)
from simulator.measurement_model import compute_sinr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _swerling_i_pd(snr_mean: float) -> float:
    """Closed-form Swerling-I P_D = exp(−η / (1 + SNR_mean))."""
    return float(np.exp(-DETECTION_THRESHOLD / (1.0 + snr_mean)))


def _z_batch(snr_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised matched-filter: z_i = |√snr_i + noise_i|², noise ~ CN(0,1)."""
    n = len(snr_array)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return np.abs(np.sqrt(snr_array) + noise) ** 2


# ---------------------------------------------------------------------------
# G5-1: Detection threshold formula
# ---------------------------------------------------------------------------

def test_detection_threshold_equals_neg_log_pfa():
    """DETECTION_THRESHOLD == −ln(P_FA) to 1e-12."""
    expected = -np.log(P_FA)
    assert abs(DETECTION_THRESHOLD - expected) < 1e-12, (
        f"DETECTION_THRESHOLD={DETECTION_THRESHOLD:.10f}, −ln(P_FA)={expected:.10f}"
    )


def test_pfa_recovered_from_threshold():
    """exp(−DETECTION_THRESHOLD) == P_FA to 1e-12."""
    assert abs(np.exp(-DETECTION_THRESHOLD) - P_FA) < 1e-12, (
        f"exp(−η)={np.exp(-DETECTION_THRESHOLD):.4e}, P_FA={P_FA}"
    )


# ---------------------------------------------------------------------------
# G5-2: Null distribution (H0, no target)
# ---------------------------------------------------------------------------

def test_null_z_mean():
    """H0: z ~ Exp(1) → E[z] = 1 (N=20 k, 3σ MC tolerance)."""
    rng = np.random.default_rng(2001)
    N = 20_000
    noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    z = np.abs(noise) ** 2                    # γ=0 case
    mc_std_of_mean = 1.0 / np.sqrt(N)        # Exp(1) has σ=1, so SE = 1/√N
    assert abs(z.mean() - 1.0) < 3 * mc_std_of_mean, (
        f"E[z]={z.mean():.5f} (expect 1.0 ± {3*mc_std_of_mean:.4f})"
    )


def test_null_z_variance():
    """H0: z ~ Exp(1) → Var[z] = 1 (N=20 k, 3σ MC tolerance)."""
    rng = np.random.default_rng(2002)
    N = 20_000
    noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    z = np.abs(noise) ** 2
    # Exp(1): μ₄=9, so Var[S²]=(μ₄−σ⁴)/N=8/N → SE=√(8/N)
    mc_std_of_var = np.sqrt(8.0 / N)
    assert abs(z.var(ddof=1) - 1.0) < 3 * mc_std_of_var, (
        f"Var[z]={z.var(ddof=1):.5f} (expect 1.0 ± {3*mc_std_of_var:.4f})"
    )


def test_null_tail_probability_formula():
    """P(z > η | H0) = exp(−η) = P_FA analytically (no MC)."""
    # For z ~ Exp(1), P(z > t) = exp(−t) exactly.
    p_fa_formula = np.exp(-DETECTION_THRESHOLD)
    assert abs(p_fa_formula - P_FA) < 1e-12


# ---------------------------------------------------------------------------
# G5-3: Swerling-I P_D formula versus Monte Carlo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snr_mean", [5.0, 20.0, 100.0])
def test_pd_swerling_i_vs_mc(snr_mean):
    """P_D formula matches Monte Carlo within 2.5% (N=50 k, Swerling-I draws).

    u ~ Exp(1), γ = snr_mean · u; z = |√γ + noise|²; detect iff z > η.
    """
    rng  = np.random.default_rng(3000 + int(snr_mean * 10))
    N    = 50_000
    u    = rng.exponential(1.0, N)          # Swerling-I RCS draw
    gamma = snr_mean * u
    z    = _z_batch(gamma, rng)
    pd_mc      = (z > DETECTION_THRESHOLD).mean()
    pd_formula = _swerling_i_pd(snr_mean)

    tol = 0.025
    assert abs(pd_mc - pd_formula) < tol, (
        f"SNR_mean={snr_mean}: P_D_mc={pd_mc:.4f}, P_D_formula={pd_formula:.4f}, "
        f"diff={abs(pd_mc - pd_formula):.4f} (tol={tol})"
    )


# ---------------------------------------------------------------------------
# G5-4: P_D monotone increasing in SNR
# ---------------------------------------------------------------------------

def test_pd_monotone_in_snr():
    """P_D = exp(−η/(1+SNR)) is strictly increasing in SNR over [0.1, 100]."""
    snr_grid = np.logspace(-1, 2, 30)
    pd_vals  = np.array([_swerling_i_pd(s) for s in snr_grid])
    diffs    = np.diff(pd_vals)
    assert np.all(diffs > 0), (
        f"P_D not monotone: first non-increasing step at SNR="
        f"{snr_grid[np.argmax(diffs <= 0)]:.2f}"
    )


def test_pd_limits():
    """P_D → P_FA as SNR→0; P_D → 1 as SNR→∞."""
    pd_lo = _swerling_i_pd(1e-6)
    pd_hi = _swerling_i_pd(1e8)
    assert abs(pd_lo - P_FA) < 1e-5, f"P_D at SNR≈0 ≠ P_FA: {pd_lo:.6e}"
    assert 1.0 - pd_hi < 1e-7,      f"P_D at SNR→∞ not ≈ 1: {pd_hi:.10f}"


# ---------------------------------------------------------------------------
# G5-5: P_D increases with SIC isolation for fixed geometry
# ---------------------------------------------------------------------------

def test_pd_increases_with_sic():
    """Higher SIC isolation → higher SINR → higher Swerling-I P_D.

    Checks all three SIC states (100, 110, 120 dB) for C1 at R=200 m,
    beam-12 boresight (az=7.5°, el=21.25°).
    """
    R, az, el, beam = 200.0, 7.5, 21.25, 12
    cfg = RESOURCE_CONFIGS['C1']

    pd_vals = []
    for sic_db in sorted(SIC_STATES_DB):
        xi  = compute_sinr(R, az, el, SIGMA_RCS_MEAN, beam, cfg, sic_db)
        snr = cfg['n_pil'] * xi          # γ_mean
        pd_vals.append(_swerling_i_pd(snr))

    for i in range(len(pd_vals) - 1):
        assert pd_vals[i] < pd_vals[i + 1], (
            f"P_D not increasing with SIC: "
            f"SIC={sorted(SIC_STATES_DB)[i]}→{sorted(SIC_STATES_DB)[i+1]} dB, "
            f"P_D={pd_vals[i]:.4f}→{pd_vals[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# G5-6: generate_detection() Monte Carlo vs analytical P_D
# ---------------------------------------------------------------------------

def test_generate_detection_pd_vs_analytical():
    """Direct MC of generate_detection() matches Swerling-I P_D within 2.5%.

    Scenario: C1, beam-12 boresight (az=7.5°, el=21.25°, R=200 m), SIC=100 dB.
    Analytical: P_D = exp(−η / (1 + γ_mean)), γ_mean = n_pil · compute_sinr(·).
    MC: N=20 000 independent draws; each call provides a fresh u_rcs ~ Exp(1).
    """
    from simulator.measurement_model import generate_detection
    from simulator.beam_steering import BEAM_CENTRES

    rng = np.random.default_rng(6001)
    cfg = RESOURCE_CONFIGS['C1']
    R, az, el, beam_b = 200.0, 7.5, 21.25, 12
    sic_db = 100.0

    # Compute analytical P_D using mean RCS
    xi_mean  = compute_sinr(R, az, el, SIGMA_RCS_MEAN, beam_b, cfg, sic_db)
    snr_mean = cfg['n_pil'] * xi_mean
    pd_analytical = float(np.exp(-DETECTION_THRESHOLD / (1.0 + snr_mean)))

    # UAV position that maps to (R, az, el) from gNB at [0,0,10]
    # az=7.5° from North (+y), el=21.25°:
    #   Δz = R·sin(el), horizontal = R·cos(el)
    #   Δx = horizontal·sin(az_rad), Δy = horizontal·cos(az_rad)
    az_r = np.radians(az)
    el_r = np.radians(el)
    horiz = R * np.cos(el_r)
    from config.sim_config import GNB_POS
    gnb = np.array(GNB_POS, dtype=float)
    target_pos = gnb + np.array([horiz * np.sin(az_r),
                                  horiz * np.cos(az_r),
                                  R * np.sin(el_r)])
    target_vel = np.zeros(3)

    N = 20_000
    detections = 0
    for _ in range(N):
        u_rcs = rng.exponential(1.0)
        result = generate_detection(target_pos, target_vel, beam_b, cfg,
                                    sic_db, u_rcs, rng, t_beam=0.0)
        if result is not None:
            detections += 1

    pd_mc = detections / N
    tol = 0.025
    assert abs(pd_mc - pd_analytical) < tol, (
        f"generate_detection() MC P_D={pd_mc:.4f}, analytical={pd_analytical:.4f}, "
        f"diff={abs(pd_mc - pd_analytical):.4f} (tol={tol})"
    )


# ---------------------------------------------------------------------------
# G5-7: Quality statistic Q_m = 10·log10(Z) on detected returns
# ---------------------------------------------------------------------------

def test_quality_statistic_above_threshold():
    """Detected returns satisfy Q_m = 10·log10(Z) > 10·log10(η) (by construction).

    generate_detection() returns quality_db = 10·log10(Z) only when Z > η.
    Collect 500 detections and verify every Q_m exceeds the threshold level.
    """
    from simulator.measurement_model import generate_detection
    from config.sim_config import GNB_POS

    rng = np.random.default_rng(7001)
    cfg = RESOURCE_CONFIGS['C4']   # high-power config → frequent detections
    R, az, el, beam_b = 150.0, 7.5, 21.25, 12
    sic_db = 120.0

    az_r, el_r = np.radians(az), np.radians(el)
    horiz = R * np.cos(el_r)
    gnb = np.array(GNB_POS, dtype=float)
    target_pos = gnb + np.array([horiz * np.sin(az_r),
                                  horiz * np.cos(az_r),
                                  R * np.sin(el_r)])
    target_vel = np.zeros(3)

    q_min_expected = 10.0 * np.log10(DETECTION_THRESHOLD)   # η threshold in dB

    collected = []
    attempts  = 0
    while len(collected) < 500 and attempts < 10_000:
        u_rcs = rng.exponential(1.0)
        result = generate_detection(target_pos, target_vel, beam_b, cfg,
                                    sic_db, u_rcs, rng, t_beam=0.0)
        attempts += 1
        if result is not None:
            _d_m, quality_db = result
            collected.append(quality_db)

    assert len(collected) >= 500, (
        f"Too few detections ({len(collected)}/500) in {attempts} attempts; "
        f"check detection probability."
    )
    q_arr = np.array(collected)
    assert np.all(q_arr > q_min_expected), (
        f"Some Q_m values ≤ threshold dB ({q_min_expected:.3f} dB): "
        f"min Q_m={q_arr.min():.3f} dB"
    )


# ---------------------------------------------------------------------------
# G5-8: Detection probability decreases with range (production pipeline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", ['C1', 'C4', 'C8'])
def test_detection_decreases_with_range_pipeline(cfg_key):
    """P_D from generate_detection() strictly decreases over R ∈ {120, 200, 300} m.

    Uses beam-12 boresight (el=21.25°) for valid UAV positions; N=5000 trials/range.
    """
    from simulator.measurement_model import generate_detection
    from config.sim_config import GNB_POS

    cfg = RESOURCE_CONFIGS[cfg_key]
    az, el, beam_b = 7.5, 21.25, 12
    sic_db = 110.0
    az_r, el_r = np.radians(az), np.radians(el)
    gnb = np.array(GNB_POS, dtype=float)
    rng = np.random.default_rng(8000 + ord(cfg_key[1]))

    ranges = [120.0, 200.0, 300.0]
    N = 5_000
    pd_vals = []
    for R in ranges:
        horiz = R * np.cos(el_r)
        target_pos = gnb + np.array([horiz * np.sin(az_r),
                                      horiz * np.cos(az_r),
                                      R * np.sin(el_r)])
        target_vel = np.zeros(3)
        hits = sum(
            1 for _ in range(N)
            if generate_detection(target_pos, target_vel, beam_b, cfg,
                                  sic_db, rng.exponential(1.0), rng,
                                  t_beam=0.0) is not None
        )
        pd_vals.append(hits / N)

    for i in range(len(pd_vals) - 1):
        assert pd_vals[i] > pd_vals[i + 1], (
            f"{cfg_key}: P_D not decreasing with range. "
            f"R={ranges[i]}→{ranges[i+1]} m: P_D={pd_vals[i]:.4f}→{pd_vals[i+1]:.4f}"
        )
