"""G2: 8×8 UPA beamforming validation.

Tests normalization identities, peak gain, gain bounds, and angular coverage.
Includes an independent closed-form array-factor verification at several directions.

Pass criteria
-------------
- ||a(az,el)||² = N_ANT = 64 for all test directions.
- ||w_b||² = 1 for all 32 beams.
- 0 ≤ G_tx ≤ 64 everywhere (floating-point tolerance).
- G_tx = 64 at each beam's own pointing direction.
- Independent array factor matches production code at sampled directions.
- Worst best-beam gain over the surveillance volume ≥ G_TX_MIN = 10.
- All 32 beam centres lie inside the intended sector.
"""

import pytest
import numpy as np

from config.sim_config import N_ANT, N_ANT_AZ, N_ANT_EL, G_TX_MIN
from simulator.beam_steering import (
    steering_vector, beam_weight, beam_gain_tx, BEAM_CENTRES, make_beam_grid,
)

# Test directions covering the surveillance volume and edges
AZ_TEST = [-52.5, -22.5, 0.0, 22.5, 52.5]
EL_TEST = [13.75, 25.0, 36.25]
TEST_DIRECTIONS = [(az, el) for az in AZ_TEST for el in EL_TEST]

TOLS = dict(
    norm  = 1e-10,   # ||a||² and ||w||²
    peak  = 1e-8,    # G_tx at beam centre
    indep = 1e-8,    # independent array factor match
)


# ---------------------------------------------------------------------------
# Independent reference: closed-form UPA array factor
# ---------------------------------------------------------------------------

def _indep_gain(az_target, el_target, az_beam, el_beam) -> float:
    """
    G_tx independently as |w_b^H a|² using the closed-form UPA array factor.

    For half-wavelength spacing:
        u = cos(el)*sin(az),  v = sin(el)
    Array factor = (1/N) × |AF_az|² × |AF_el|²
        AF_az = Σ_{m=0}^{N_AZ-1} exp(jπ(u_b−u_t)m)
        AF_el = Σ_{n=0}^{N_EL-1} exp(jπ(v_b−v_t)n)
    """
    def to_uv(az, el):
        return (np.cos(np.radians(el)) * np.sin(np.radians(az)),
                np.sin(np.radians(el)))

    u_t, v_t = to_uv(az_target, el_target)
    u_b, v_b = to_uv(az_beam,   el_beam)

    m = np.arange(N_ANT_AZ)
    n = np.arange(N_ANT_EL)
    AF_az = np.sum(np.exp(1j * np.pi * (u_b - u_t) * m))
    AF_el = np.sum(np.exp(1j * np.pi * (v_b - v_t) * n))
    return float(np.abs(AF_az) ** 2 * np.abs(AF_el) ** 2 / N_ANT)


# ---------------------------------------------------------------------------
# G2-1: Steering vector norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az,el", TEST_DIRECTIONS)
def test_steering_vector_norm(az, el):
    """||a(az,el)||² = N_ANT = 64 for all test directions."""
    a = steering_vector(az, el)
    norm_sq = float(np.real(a @ np.conj(a)))
    assert abs(norm_sq - N_ANT) < TOLS['norm'], (
        f"||a||²={norm_sq:.10f} at az={az}, el={el}"
    )


# ---------------------------------------------------------------------------
# G2-2: Beam weight vector norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", range(32))
def test_beam_weight_norm(b):
    """||w_b||² = 1 for all 32 beams."""
    az_b, el_b = BEAM_CENTRES[b]
    w = beam_weight(az_b, el_b)
    norm_sq = float(np.real(w @ np.conj(w)))
    assert abs(norm_sq - 1.0) < TOLS['norm'], (
        f"beam {b}: ||w_b||²={norm_sq:.12f}"
    )


# ---------------------------------------------------------------------------
# G2-3: Peak gain at beam centre
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", range(32))
def test_peak_gain_at_beam_centre(b):
    """G_tx = 64 when target direction matches beam b's pointing direction."""
    az_b, el_b = BEAM_CENTRES[b]
    g = beam_gain_tx(az_b, el_b, BEAM_CENTRES, b)
    assert abs(g - N_ANT) < TOLS['peak'], (
        f"beam {b}: peak gain={g:.10f}, expected {N_ANT}"
    )


# ---------------------------------------------------------------------------
# G2-4: Gain bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az,el", TEST_DIRECTIONS)
@pytest.mark.parametrize("b", [0, 8, 16, 24])
def test_gain_bounds(az, el, b):
    """0 ≤ G_tx ≤ 64 for arbitrary directions and beams."""
    g = beam_gain_tx(az, el, BEAM_CENTRES, b)
    assert -TOLS['norm'] <= g <= N_ANT + TOLS['norm'], (
        f"G_tx={g:.4f} at az={az}, el={el}, beam {b}"
    )


# ---------------------------------------------------------------------------
# G2-5: Independent array factor verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az,el", TEST_DIRECTIONS)
@pytest.mark.parametrize("b", [0, 4, 15, 31])
def test_independent_array_factor(az, el, b):
    """G_tx matches the closed-form UPA array factor (no production code)."""
    az_b, el_b = BEAM_CENTRES[b]
    g_sim = beam_gain_tx(az, el, BEAM_CENTRES, b)
    g_ref = _indep_gain(az, el, az_b, el_b)
    assert abs(g_sim - g_ref) < TOLS['indep'], (
        f"beam {b}, az={az}, el={el}: sim={g_sim:.8f}, ref={g_ref:.8f}"
    )


# ---------------------------------------------------------------------------
# G2-6: Beam grid structure
# ---------------------------------------------------------------------------

def test_beam_grid_has_32_entries():
    """BEAM_CENTRES contains exactly 32 entries."""
    assert len(BEAM_CENTRES) == 32


def test_beam_grid_covers_intended_sector():
    """All beam centres lie inside the intended surveillance sector."""
    for b, (az, el) in enumerate(BEAM_CENTRES):
        assert -60.0 <= az <= 60.0,  f"Beam {b}: az={az:.2f}° outside ±60°"
        assert  10.0 <= el <= 40.0,  f"Beam {b}: el={el:.2f}° outside [10°,40°]"


def test_beam_centres_match_make_beam_grid():
    """Module-level BEAM_CENTRES equals make_beam_grid() output."""
    fresh = make_beam_grid()
    assert len(fresh) == len(BEAM_CENTRES)
    for b, ((az1, el1), (az2, el2)) in enumerate(zip(BEAM_CENTRES, fresh)):
        assert abs(az1 - az2) < 1e-9 and abs(el1 - el2) < 1e-9, (
            f"Beam {b}: BEAM_CENTRES=({az1},{el1}), make_beam_grid=({az2},{el2})"
        )


# ---------------------------------------------------------------------------
# G2-7: Best-beam coverage over the surveillance volume
# ---------------------------------------------------------------------------

def test_best_beam_coverage():
    """Worst-case best-beam gain over the surveillance volume ≥ G_TX_MIN = 10."""
    az_grid = np.linspace(-60, 60, 61)
    el_grid = np.linspace(10, 40, 31)
    min_best = np.inf
    for az in az_grid:
        for el in el_grid:
            best = max(beam_gain_tx(az, el, BEAM_CENTRES, b) for b in range(32))
            if best < min_best:
                min_best = best
    assert min_best >= G_TX_MIN, (
        f"Worst best-beam gain={min_best:.3f} < G_TX_MIN={G_TX_MIN}"
    )
