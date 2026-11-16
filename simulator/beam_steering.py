"""
simulator/beam_steering.py
==========================
UPA beam-steering model for the RCATT ISAC simulator.

The gNB employs an idealized, untilted 8×8 vertical UPA (N_ANT=64 elements)
with physical boresight at azimuth=0°, elevation=0°.  Mechanical antenna
tilt and element-pattern effects are not modeled.  32 pre-defined beams
electronically steer over the surveillance volume (az ∈ [−60°,60°], el ∈ [10°,40°]).

Public API
----------
make_beam_grid()        → list of 32 (az_deg, el_deg) beam-centre pairs
steering_vector(az, el) → complex array, shape (64,)
beam_gain_tx(az, el, beam_centres, b) → scalar G_tx (linear, ≤ 64)

All angles in degrees unless noted.  Array element ordering: row-major
(elevation index varies slowest, azimuth index fastest).
"""

import numpy as np
from config.sim_config import (
    N_ANT_AZ, N_ANT_EL, N_ANT, LAMBDA,
    SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
    SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
    N_BEAMS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deg2rad(x):
    return np.asarray(x, dtype=float) * (np.pi / 180.0)


def _azel_to_uvw(az_deg, el_deg):
    """Convert (az, el) to direction cosines (u, v, w).

    Convention (standard radar / antenna):
        u = cos(el) * sin(az)
        v = sin(el)
        w = cos(el) * cos(az)   (unused in 2-D steering but kept for reference)
    """
    az = _deg2rad(az_deg)
    el = _deg2rad(el_deg)
    u = np.cos(el) * np.sin(az)
    v = np.sin(el)
    return u, v


# ---------------------------------------------------------------------------
# Steering vector
# ---------------------------------------------------------------------------
def steering_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit-modulus UPA steering vector  a(θ,φ) ∈ ℂ^{N_ANT}.

    Half-wavelength element spacing (d = λ/2) in both dimensions.
    Phase reference: element (0,0) at the array origin.

    Layout: a[el_idx * N_ANT_AZ + az_idx] for
        el_idx ∈ {0,…,N_ANT_EL-1}, az_idx ∈ {0,…,N_ANT_AZ-1}.

    Parameters
    ----------
    az_deg, el_deg : float
        Target azimuth and elevation in degrees.

    Returns
    -------
    a : ndarray, shape (N_ANT,), dtype complex128
        Unit-modulus steering vector; each entry has |a_n| = 1.
    """
    u, v = _azel_to_uvw(az_deg, el_deg)
    # Phase increments (in radians) per element in each axis
    # Δφ_az = π u,  Δφ_el = π v   (d = λ/2  →  2πd/λ = π)
    az_idx = np.arange(N_ANT_AZ)         # 0,1,...,7
    el_idx = np.arange(N_ANT_EL)         # 0,1,...,7
    phase_az = np.pi * u * az_idx        # shape (N_ANT_AZ,)
    phase_el = np.pi * v * el_idx        # shape (N_ANT_EL,)
    # Outer sum → (N_ANT_EL, N_ANT_AZ), then flatten row-major
    phase = phase_el[:, None] + phase_az[None, :]   # (8, 8)
    return np.exp(1j * phase).ravel()               # (64,)


# ---------------------------------------------------------------------------
# Beam weight vector (beam b points at beam_centre[b])
# ---------------------------------------------------------------------------
def beam_weight(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit-modulus MRT weight vector for a beam pointed at (az_deg, el_deg).

    w_b = a*(θ_b, φ_b) / sqrt(N_ANT)   →  ||w_b||² = 1.
    """
    return np.conj(steering_vector(az_deg, el_deg)) / np.sqrt(N_ANT)


def beam_gain_tx(az_deg: float, el_deg: float,
                 beam_centres: list, b: int) -> float:
    """Transmit array gain G_tx for a target at (az_deg, el_deg) when
    beam b is active.

    G_tx = |w_b^H a(θ,φ)|²

    Maximum value is N_ANT = 64 (perfect alignment).
    Unit-modulus weights → G_tx ∈ [0, 64].

    Parameters
    ----------
    az_deg, el_deg  : target direction
    beam_centres    : list of (az, el) tuples, length N_BEAMS
    b               : beam index (0-based)
    """
    az_b, el_b = beam_centres[b]
    w_b = beam_weight(az_b, el_b)              # (64,) conjugated / normalised
    a   = steering_vector(az_deg, el_deg)      # (64,)
    return float(np.abs(w_b @ a) ** 2)


# ---------------------------------------------------------------------------
# Beam grid
# ---------------------------------------------------------------------------
def make_beam_grid() -> list:
    """Return a list of N_BEAMS (az_deg, el_deg) beam-centre pairs that
    uniformly tile the surveillance volume.

    Grid layout: N_BEAMS_AZ × N_BEAMS_EL beams with equal angular spacing.
    For N_BEAMS=32 we use 8 azimuth × 4 elevation.

    Azimuth  : 8 beams spanning [−60°, +60°]  → centres at ±7.5°, ±22.5°, ±37.5°, ±52.5°
    Elevation: 4 beams spanning [10°,  40°]   → centres at 13.75°, 21.25°, 28.75°, 36.25°
    """
    assert N_BEAMS == 32, "make_beam_grid assumes N_BEAMS=32 (8 az × 4 el)"
    n_az, n_el = 8, 4
    az_centres = np.linspace(SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
                             n_az, endpoint=False) + \
                 (SURV_AZ_MAX_DEG - SURV_AZ_MIN_DEG) / (2 * n_az)
    el_centres = np.linspace(SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
                             n_el, endpoint=False) + \
                 (SURV_EL_MAX_DEG - SURV_EL_MIN_DEG) / (2 * n_el)
    grid = [(float(az), float(el))
            for el in el_centres
            for az in az_centres]
    return grid   # length 32, ordered el-major


# ---------------------------------------------------------------------------
# Module-level beam grid (created once on import)
# ---------------------------------------------------------------------------
BEAM_CENTRES = make_beam_grid()   # list of 32 (az_deg, el_deg) tuples
