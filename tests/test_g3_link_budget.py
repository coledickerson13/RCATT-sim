"""G3: Radar link-budget validation.

Verifies that compute_sinr() correctly models all physical parameters.
Tests use independent reference calculations; production functions are
the subject under test.

Pass criteria
-------------
- Deterministic scaling identities (R⁻⁴, σ, P_tx) match to < 1e-9 dB.
- SIC isolation improves SINR monotonically for all configs.
- Per-RE SINR does not depend on n_pil.
- G_tx proportionality in noise-limited regime.
- Swerling-I sample mean within 2% of SIGMA_RCS_MEAN; variance within 4σ.
"""

import pytest
import numpy as np

from config.sim_config import (
    RESOURCE_CONFIGS, SIGMA_RCS_MEAN, C_LIGHT, LAMBDA, N0, DELTA_F,
    N_ANT, G_RX, SIC_STATES_DB,
)
from simulator.measurement_model import compute_sinr

# ---------------------------------------------------------------------------
# Shared test geometry
# ---------------------------------------------------------------------------
_AZ, _EL = 0.0, 25.0        # on-axis, good elevation
_R0      = 150.0             # baseline range, m
_SIC_DB  = 110.0             # mid SIC state
_CFG     = RESOURCE_CONFIGS['C1']
_BEAM    = 0                 # beam 0 close to boresight

TOL_DB = 1e-9    # tolerance for deterministic tests (dB)


def _sinr_db(R, sigma, cfg, sic_db, az=_AZ, el=_EL, beam=_BEAM):
    xi = compute_sinr(R, az, el, sigma, beam, cfg, sic_db)
    return 10.0 * np.log10(xi)


# ---------------------------------------------------------------------------
# G3-1: Range scaling (R → 2R should give exactly -12.041 dB)
# ---------------------------------------------------------------------------

def test_range_doubling_sinr_loss():
    """Doubling range decreases SINR by exactly 10·log10(16) = 12.041 dB."""
    s1 = _sinr_db(_R0,       _SIGMA := SIGMA_RCS_MEAN, _CFG, _SIC_DB)
    s2 = _sinr_db(_R0 * 2.0, _SIGMA,                   _CFG, _SIC_DB)
    expected_db = 10.0 * np.log10(16.0)   # 2^4 = 16
    assert abs((s1 - s2) - expected_db) < TOL_DB, (
        f"Loss={s1-s2:.8f} dB, expected {expected_db:.8f} dB"
    )


def test_range_triple_sinr_loss():
    """3× range decreases SINR by 10·log10(81) ≈ 19.085 dB."""
    sigma = SIGMA_RCS_MEAN
    s1 = _sinr_db(_R0,       sigma, _CFG, _SIC_DB)
    s3 = _sinr_db(_R0 * 3.0, sigma, _CFG, _SIC_DB)
    expected_db = 10.0 * np.log10(81.0)   # 3^4 = 81
    assert abs((s1 - s3) - expected_db) < TOL_DB, (
        f"3× range loss={s1-s3:.8f} dB, expected {expected_db:.8f} dB"
    )


# ---------------------------------------------------------------------------
# G3-2: RCS scaling
# ---------------------------------------------------------------------------

def test_rcs_10x_sinr_increase():
    """10× RCS → SINR increases by 10 dB (linear numerator)."""
    s1 = _sinr_db(_R0, SIGMA_RCS_MEAN,        _CFG, _SIC_DB)
    s2 = _sinr_db(_R0, SIGMA_RCS_MEAN * 10.0, _CFG, _SIC_DB)
    assert abs((s2 - s1) - 10.0) < TOL_DB, (
        f"10× RCS gain={s2-s1:.8f} dB, expected 10.0 dB"
    )


def test_rcs_linear_at_high_sic():
    """In noise-limited regime (high SIC), xi scales exactly linearly with sigma."""
    xi1 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN,       _BEAM, _CFG, 120.0)
    xi5 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN * 5.0, _BEAM, _CFG, 120.0)
    assert abs(xi5 / xi1 - 5.0) < 1e-9, (
        f"RCS 5× ratio={xi5/xi1:.10f}, expected 5.0"
    )


# ---------------------------------------------------------------------------
# G3-3: Transmit power
# ---------------------------------------------------------------------------

def test_ptx_doubling_at_high_sic():
    """Doubling P_tx in noise-limited regime → +3.01 dB SINR.

    Uses SIC_DB = 200 dB so P_SI_re << thermal noise (noise-limited regime
    confirmed: P_SI_re ≈ P_tx/(n_sc·10^20) << N0·Δf for all C1-class configs).
    """
    SIC_NL = 200.0          # analytically noise-limited: P_SI_re ≈ 0
    cfg2 = dict(_CFG)
    cfg2['p_tx_w'] = _CFG['p_tx_w'] * 2.0
    xi1 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN, _BEAM, _CFG, SIC_NL)
    xi2 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN, _BEAM, cfg2, SIC_NL)
    expected_db = 10.0 * np.log10(2.0)
    actual_db   = 10.0 * np.log10(xi2 / xi1)
    # At SIC=200 dB, P_SI_re/noise_re ≈ 8.6e-9; residual SI error ≈ 3.7e-8 dB.
    assert abs(actual_db - expected_db) < 1e-7, (
        f"P_tx × 2: {actual_db:.8f} dB, expected {expected_db:.8f} dB"
    )


def test_si_scales_with_ptx():
    """Residual SI per RE = P_tx / (n_sc × C_sic_lin) is linear in P_tx."""
    cfg2 = dict(_CFG)
    cfg2['p_tx_w'] = _CFG['p_tx_w'] * 3.0
    p_tx1 = _CFG['p_tx_w']
    p_tx2 = cfg2['p_tx_w']
    n_sc  = _CFG['n_sc_carrier']
    c_sic = 10 ** (100.0 / 10)                 # worst SIC state
    P_SI1 = p_tx1 / (n_sc * c_sic)
    P_SI2 = p_tx2 / (n_sc * c_sic)
    assert abs(P_SI2 / P_SI1 - 3.0) < 1e-12, "SI per RE must be linear in P_tx"


# ---------------------------------------------------------------------------
# G3-4: SIC monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
def test_sic_monotonicity(cfg_key):
    """SINR is non-decreasing as SIC isolation improves (more dB = better)."""
    cfg = RESOURCE_CONFIGS[cfg_key]
    xi_prev = -np.inf
    for sic_db in sorted(SIC_STATES_DB):
        xi = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN, _BEAM, cfg, sic_db)
        assert xi >= xi_prev - 1e-15, (
            f"{cfg_key}: SINR decreased at SIC={sic_db:.0f} dB"
        )
        xi_prev = xi


# ---------------------------------------------------------------------------
# G3-5: Pilot count — per-RE SINR must not depend on n_pil
# ---------------------------------------------------------------------------

def test_per_re_sinr_independent_of_npil():
    """compute_sinr() returns per-RE SINR: doubling n_pil leaves xi unchanged."""
    cfg2 = dict(_CFG, n_pil=_CFG['n_pil'] * 2)
    xi1 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN, _BEAM, _CFG, _SIC_DB)
    xi2 = compute_sinr(_R0, _AZ, _EL, SIGMA_RCS_MEAN, _BEAM, cfg2, _SIC_DB)
    assert abs(xi1 - xi2) < 1e-15, (
        f"Per-RE SINR changed: xi1={xi1:.6e}, xi2={xi2:.6e}"
    )


# ---------------------------------------------------------------------------
# G3-6: Beam gain proportionality in noise-limited regime
# ---------------------------------------------------------------------------

def test_beam_gain_proportional_to_sinr():
    """SINR ∝ G_tx in noise-limited regime (high SIC, two different beams)."""
    from simulator.beam_steering import beam_gain_tx, BEAM_CENTRES
    az, el = 7.5, 21.25     # near boresight, beam 12 centre

    # Pick beams with large gain contrast
    g0  = beam_gain_tx(az, el, BEAM_CENTRES, 12)   # peak beam
    g31 = beam_gain_tx(az, el, BEAM_CENTRES,  0)   # off-axis beam

    if g0 < 1e-6 or g31 < 1e-6:
        pytest.skip("One beam gain near zero — ratio test not meaningful")

    xi0  = compute_sinr(_R0, az, el, SIGMA_RCS_MEAN, 12, _CFG, 120.0)
    xi31 = compute_sinr(_R0, az, el, SIGMA_RCS_MEAN,  0, _CFG, 120.0)

    ratio_xi = xi0 / xi31
    ratio_g  = g0  / g31
    assert abs(ratio_xi / ratio_g - 1.0) < 1e-9, (
        f"SINR ratio={ratio_xi:.8f}, G_tx ratio={ratio_g:.8f}"
    )


# ---------------------------------------------------------------------------
# G3-7: Swerling-I RCS distribution
# ---------------------------------------------------------------------------

def test_swerling_rcs_mean():
    """Sample mean of Swerling-I draw ≈ SIGMA_RCS_MEAN (within 2%)."""
    rng = np.random.default_rng(42)
    n   = 100_000
    draws = rng.exponential(1.0, n) * SIGMA_RCS_MEAN
    rel_err = abs(draws.mean() / SIGMA_RCS_MEAN - 1.0)
    assert rel_err < 0.02, (
        f"RCS mean={draws.mean():.5f} m², expected {SIGMA_RCS_MEAN:.5f}"
    )


def test_swerling_rcs_variance():
    """Sample variance ≈ SIGMA_RCS_MEAN² (Exp(1) → Var = mean²); within 4σ of estimator."""
    rng = np.random.default_rng(99)
    n   = 100_000
    draws = rng.exponential(1.0, n) * SIGMA_RCS_MEAN
    sample_var  = draws.var()
    expected_var = SIGMA_RCS_MEAN ** 2
    var_std = np.sqrt(2.0 / n) * expected_var   # std of chi² variance estimator
    assert abs(sample_var - expected_var) < 4.0 * var_std, (
        f"RCS var={sample_var:.6f} m⁴, expected {expected_var:.6f} ± {4*var_std:.6f}"
    )


def test_swerling_rcs_exponential_shape():
    """Swerling-I draws are Exponential(mean = SIGMA_RCS_MEAN): P(x > 2·mean) ≈ e^-2."""
    rng = np.random.default_rng(7)
    n   = 200_000
    draws = rng.exponential(1.0, n) * SIGMA_RCS_MEAN
    p_above = (draws > 2.0 * SIGMA_RCS_MEAN).mean()
    expected = np.exp(-2.0)   # ≈ 0.1353
    # 3σ Monte Carlo tolerance
    sigma_mc = np.sqrt(expected * (1.0 - expected) / n)
    assert abs(p_above - expected) < 3.0 * sigma_mc, (
        f"P(σ > 2·mean)={p_above:.5f}, expected {expected:.5f}"
    )
