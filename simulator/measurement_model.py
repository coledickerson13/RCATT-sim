"""
RCATT Simulator — Measurement Model
====================================
Implements the full detection pipeline for one beam dwell:

  target Cartesian state
      → (R, az, el, v_r)                           [state_to_polar]
      → per-pilot SINR ξ (mean RCS + Swerling draw) [compute_sinr]
      → CRLB noise variances                        [compute_crlb_variances]
      → R_sim = κ·R_CRLB + diag(σ²_floor)          [compute_r_sim]
      → Ricean detection + quality feature Z          [generate_detection]
      → d_m ∈ ℝ⁷ or None

All angles in degrees externally; radians used only inside computations.
Measurement vector d_m: [R̃, ṽ_r, θ̃_az, φ̃_el, θ_b, φ_b, δt_m]
  R̃      – noisy range (m)
  ṽ_r    – noisy radial velocity (m/s, positive = receding)
  θ̃_az  – noisy azimuth (degrees)
  φ̃_el  – noisy elevation (degrees)
  θ_b    – beam-centre azimuth (degrees)
  φ_b    – beam-centre elevation (degrees)
  δt_m   – time offset of this beam within the scan (s)

References
----------
  v3 RCATT Measurement Generation Reference (authoritative parameter set)
  Fernandez et al., arXiv 2603.28318v4 (2026) — Full FIM/CRLB for 5G NR OFDM sensing
  Abu-Shaban et al., IEEE TWC 2018 (CRLB for OFDM delay/Doppler/angle)
  Swerling (1954); Mahafza (2013) ch.6 (Swerling-I detection probability)
"""
from __future__ import annotations

import numpy as np

from config.sim_config import (
    C_LIGHT, LAMBDA, DELTA_F, N0,
    N_ANT_AZ, N_ANT_EL,
    G_RX,
    SIGMA_RCS_MEAN,
    GNB_POS,
    KAPPA,
    FLOOR_RANGE_M, FLOOR_VEL_MPS, FLOOR_AZ_DEG, FLOOR_EL_DEG,
    DETECTION_THRESHOLD,
)
from simulator.beam_steering import BEAM_CENTRES, beam_gain_tx
from simulator.geometry import target_in_fov


# ---------------------------------------------------------------------------
# 1.  Coordinate conversion
# ---------------------------------------------------------------------------

def state_to_polar(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    gnb_pos: np.ndarray = GNB_POS,
) -> tuple[float, float, float, float]:
    """Convert Cartesian target state to polar sensing coordinates.

    Parameters
    ----------
    target_pos : (3,) array [x, y, z] in metres
    target_vel : (3,) array [vx, vy, vz] in m/s
    gnb_pos    : (3,) gNB position, default from sim_config

    Returns
    -------
    R      : slant range (m), always > 0
    az_deg : azimuth (degrees) — atan2(Δx, Δy), 0° = North (+y); matches scene.py birth convention
    el_deg : elevation (degrees) — arcsin(Δz / R), 0° = horizon
    vr     : radial velocity (m/s), positive = moving away from gNB
    """
    delta = target_pos - gnb_pos            # vector from gNB to target
    R = float(np.linalg.norm(delta))
    if R < 1e-6:                            # guard: target exactly at gNB
        return 0.0, 0.0, 0.0, 0.0

    unit = delta / R
    az_deg = float(np.degrees(np.arctan2(delta[0], delta[1])))  # 0°=North(+y)
    el_deg = float(np.degrees(np.arcsin(np.clip(delta[2] / R, -1.0, 1.0))))
    vr = float(np.dot(target_vel, unit))    # radial component
    return R, az_deg, el_deg, vr


# ---------------------------------------------------------------------------
# 2.  Per-pilot SINR
# ---------------------------------------------------------------------------

def compute_sinr(
    R: float,
    az_deg: float,
    el_deg: float,
    sigma_rcs: float,
    beam_b: int,
    cfg: dict,
    sic_state_db: float,
) -> float:
    """Compute the per-RE SINR for one target.

    Model (Reference v3)
    --------------------
    Transmit power is allocated uniformly across all N_sc_carrier resource
    elements of the carrier (not just PRS pilots), so each RE carries:
        p_re = P_tx / N_sc_carrier

    Two-way echo power per RE:
        P_rx_re = p_re · G_tx · G_rx · λ² · σ / ((4π)³ · R⁴)

    Residual self-interference per RE:
        P_SI_re = P_tx / (N_sc_carrier · C_SIC_lin)

    Per-RE SINR:
        ξ = P_rx_re / (N₀ · Δf + P_SI_re)

    Post-integration SNR (after coherent combining of N_pil pilots):
        γ = N_pil · ξ   [computed by the caller]

    Parameters
    ----------
    R           : range (m)
    az_deg, el_deg : target angles
    sigma_rcs   : target RCS draw (m²); for mean use SIGMA_RCS_MEAN
    beam_b      : beam index (0-31)
    cfg         : resource-config dict from RESOURCE_CONFIGS
    sic_state_db: SIC isolation (dB)

    Returns
    -------
    xi : per-RE SINR (linear); multiply by n_pil for post-integration SNR
    """
    P_tx         = cfg['p_tx_w']
    n_sc_carrier = cfg['n_sc_carrier']

    G_tx = beam_gain_tx(az_deg, el_deg, BEAM_CENTRES, beam_b)

    # Two-way radar channel gain (Friis monostatic)
    path_gain = G_tx * G_RX * (LAMBDA ** 2) * sigma_rcs / ((4 * np.pi) ** 3 * R ** 4)

    p_per_re   = P_tx / n_sc_carrier           # transmit power per RE (W)
    P_rx_re    = p_per_re * path_gain          # echo power per RE (W)

    C_sic_lin  = 10 ** (sic_state_db / 10)
    P_SI_re    = P_tx / (n_sc_carrier * C_sic_lin)  # SI per RE (W)

    noise_re   = N0 * DELTA_F                  # thermal noise per RE (W)

    xi = P_rx_re / (noise_re + P_SI_re)
    return float(xi)


# ---------------------------------------------------------------------------
# 3.  CRLB noise variances
# ---------------------------------------------------------------------------

def compute_crlb_variances(
    xi: float,
    cfg: dict,
    az_deg: float,
    el_deg: float,
) -> dict[str, float]:
    """Cramér-Rao lower bounds for the four measurement dimensions.

    Uses the per-pilot SINR ξ; coherent integration of N_pil pilots gives
    total post-MF SNR  SNR_coh = N_pil · ξ.

    Range CRLB
    ----------
    Pilots are on a Comb-2 grid: N_sc_PRS = 6·N_RB_PRS subcarriers per
    symbol with spacing 2·Δf.  RMS bandwidth of that grid:
        β² = (N_sc_PRS² − 1)/12 · (2Δf)²   (uniform frequency samples)

    Monostatic round-trip: effective delay τ = 2R/c, so ∂τ/∂R = 2/c.
        σ²_R = 1 / ((2/c)² · 8π² · SNR_coh · β²)
             = c² / (32π² · SNR_coh · β²)

    Doppler / radial-velocity CRLB
    --------------------------------
    PRS symbols occupy positions k ∈ {0,1,2,3} of each slot.  The exact
    pilot-time variance σ_t² is computed from these positions (not the
    uniform approximation T_obs²/12, which under-estimates σ_t by ~10%).

        σ²_vr = λ² / (32π² · SNR_coh · σ_t²)

    Angle CRLBs (separable UPA, N_AZ × N_EL, half-wavelength spacing)
    -------------------------------------------------------------------
    Direction cosines: u = cos(el)·sin(az),  v = sin(el).

    For the separable rectangular UPA the (u, v) cross-term of the Fisher
    information matrix vanishes (J_uv = 0).  The diagonal elements are:
        J_uu = SNR_coh · π² · (N_AZ²−1) / 6
        J_vv = SNR_coh · π² · (N_EL²−1) / 6
    giving:
        σ²_u = 6 / (SNR_coh · π² · (N_AZ²−1))
        σ²_v = 6 / (SNR_coh · π² · (N_EL²−1))

    Jacobian J maps d(az,el) → d(u,v):
        ∂u/∂az = cos(el)·cos(az),  ∂u/∂el = −sin(el)·sin(az)
        ∂v/∂az = 0,                ∂v/∂el =  cos(el)

    Because ∂u/∂el ≠ 0, elevation uncertainty contributes to azimuth
    uncertainty (u depends on both angles).  The joint angular covariance
    is C_angles = J⁻¹ · diag(σ²_u, σ²_v) · J⁻¹ᵀ; only the diagonal
    variances enter R_sim, keeping the measurement model diagonal.

    Returns
    -------
    dict with keys 'R'(m²), 'vr'((m/s)²), 'az'(deg²), 'el'(deg²)
    """
    n_pil    = cfg['n_pil']
    n_rb_prs = cfg['n_rb_prs']
    t_obs    = cfg['t_obs_s']

    SNR_coh = n_pil * xi           # total post-integration SINR

    # --- range ---
    n_sc_prs = 6 * n_rb_prs        # Comb-2: 6 pilot sc per RB per symbol
    beta_sq  = (n_sc_prs ** 2 - 1) / 12.0 * (2 * DELTA_F) ** 2
    var_R    = C_LIGHT ** 2 / (32 * np.pi ** 2 * SNR_coh * beta_sq)

    # --- radial velocity ---
    # Exact PRS symbol times: symbols k=0,1,2,3 within each of n_slot slots.
    # T_sym = cfg['cpi_s']/cfg['n_sym'] = SLOT_DURATION/N_SYM_PER_SLOT.
    _T_sym   = cfg['cpi_s'] / cfg['n_sym']
    _n_slot  = cfg['n_slot']
    _times   = np.array([(14 * s + k) * _T_sym
                         for s in range(_n_slot) for k in range(4)])
    _sigma_t_sq = float(np.var(_times))           # ddof=0
    var_vr = LAMBDA ** 2 / (32.0 * np.pi ** 2 * SNR_coh * _sigma_t_sq)

    # --- azimuth (direction cosine u = cos(el)·sin(az)) ---
    # Separable UPA: J_uv = 0 analytically (orthogonal aperture dimensions).
    # J_uu = SNR_coh·π²·(N_AZ²-1)/6  →  σ²_u = 6/(SNR_coh·π²·(N_AZ²-1))
    # Jacobian du/daz = cos(el)·cos(az) gives the pointing-direction dependence.
    el_r  = np.radians(el_deg)
    az_r  = np.radians(az_deg)
    cos_el = np.cos(el_r)
    cos_az = np.cos(az_r)

    sigma_u_sq = 6.0 / (SNR_coh * np.pi ** 2 * (N_ANT_AZ ** 2 - 1))
    sigma_v_sq = 6.0 / (SNR_coh * np.pi ** 2 * (N_ANT_EL ** 2 - 1))

    # --- elevation & azimuth: 2×2 Jacobian inversion ---
    # J[i,j] = ∂(u,v)_i / ∂(az,el)_j
    # ∂u/∂el = −sin(el)·sin(az) is the cross-term that mixes σ²_v into var_az.
    sin_el = np.sin(el_r)
    sin_az = np.sin(az_r)
    J = np.array([[cos_el * cos_az, -sin_el * sin_az],
                  [0.0,              cos_el           ]])
    J_inv    = np.linalg.inv(J)
    C_angles = J_inv @ np.diag([sigma_u_sq, sigma_v_sq]) @ J_inv.T
    var_az_deg = C_angles[0, 0] * (180.0 / np.pi) ** 2
    var_el_deg = C_angles[1, 1] * (180.0 / np.pi) ** 2

    return {'R': var_R, 'vr': var_vr, 'az': var_az_deg, 'el': var_el_deg}


# ---------------------------------------------------------------------------
# 4.  R_sim = κ · R_CRLB + diag(σ²_floor)
# ---------------------------------------------------------------------------

def compute_r_sim(crlb_vars: dict[str, float]) -> np.ndarray:
    """Build the 4×4 diagonal simulation noise covariance.

    R_sim = κ · R_CRLB + diag([σ²_R_floor, σ²_vr_floor, σ²_az_floor, σ²_el_floor])

    where κ = KAPPA accounts for processing losses (mis-matched filter,
    quantisation, calibration errors …).

    Returns
    -------
    R_sim : (4, 4) diagonal ndarray  [R (m²), vr ((m/s)²), az (deg²), el (deg²)]
    """
    crlb_diag = np.array([
        crlb_vars['R'],
        crlb_vars['vr'],
        crlb_vars['az'],
        crlb_vars['el'],
    ])
    floor_sq = np.array([
        FLOOR_RANGE_M ** 2,
        FLOOR_VEL_MPS ** 2,
        FLOOR_AZ_DEG ** 2,
        FLOOR_EL_DEG ** 2,
    ])
    return np.diag(KAPPA * crlb_diag + floor_sq)


# ---------------------------------------------------------------------------
# 5.  Full detection pipeline for one beam dwell
# ---------------------------------------------------------------------------

def generate_detection(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    beam_b: int,
    cfg: dict,
    sic_state_db: float,
    u_rcs: float,
    rng: np.random.Generator,
    t_beam: float,
) -> tuple[np.ndarray, float] | None:
    """Run the full detection pipeline for one (target, beam, config) triple.

    Parameters
    ----------
    target_pos   : (3,) Cartesian position [x, y, z] (m)
    target_vel   : (3,) velocity [vx, vy, vz] (m/s)
    beam_b       : beam index 0–31
    cfg          : resource-config dict (from RESOURCE_CONFIGS)
    sic_state_db : SIC isolation (dB)
    u_rcs        : Swerling-I RCS draw ~ Exp(1); σ_RCS = u_rcs · SIGMA_RCS_MEAN
    rng          : numpy Generator for reproducibility
    t_beam       : δt_m = t_beam - t_k (s); always ≤ 0, relative to reporting boundary t_k; becomes δt_m in d_m

    Returns
    -------
    (d_m, sinr_obs_db) : tuple, or None if the target is not detected.
      d_m          : (7,) ndarray = [R̃, ṽ_r, θ̃_az, φ̃_el, θ_b, φ_b, δt_m]
      quality_db   : 10·log10(Z), where Z = |√γ_actual + n|² is the per-frame
                     Ricean matched-filter power (Swerling-I: fixed amplitude
                     from xi_actual draw, unknown carrier phase, CN(0,1) noise).
                     Appended by the caller as the 8th feature Q_m.

    Notes
    -----
    Detection and Q_m use the same Ricean observation Z = |√γ_actual + n|²
    (CN noise rotation-invariance makes the carrier phase draw unnecessary).
    Deep RCS fades → low γ_actual → Z near noise floor → lower P_D and Q_m.
    Fixed threshold DETECTION_THRESHOLD gives PFA = exp(−η) = 1e-4.
    Clutter Q_m uses DETECTION_THRESHOLD + Exp(1) (memoryless; scene.py).
    CRLB covariances use the realised xi_actual (actual RCS draw this frame).
    """
    # --- geometry ---
    R, az_deg, el_deg, vr = state_to_polar(target_pos, target_vel)

    # Target outside the modeled surveillance region — skip
    if not target_in_fov(target_pos):
        return None

    # --- SINR: actual RCS draw drives detection and Q_m ---
    sigma_rcs_actual = u_rcs * SIGMA_RCS_MEAN
    xi_actual        = compute_sinr(R, az_deg, el_deg, sigma_rcs_actual,
                                    beam_b, cfg, sic_state_db)
    snr_actual = cfg['n_pil'] * xi_actual  # post-integration SNR (this frame)

    # --- detection + quality: Ricean matched-filter observation (Swerling-I per frame) ---
    # Fixed amplitude √γ_actual from xi_actual draw; carrier phase unknown.
    # CN noise rotation-invariance means e^{jφ}·n ~ CN(0,1), so the phase
    # draw is unnecessary: Z = |√γ_actual + n|², n ~ CN(0,1).
    noise       = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2.0)
    z_obs       = abs(np.sqrt(snr_actual) + noise) ** 2
    if z_obs <= DETECTION_THRESHOLD:      # fixed threshold, PFA = exp(−DETECTION_THRESHOLD) = 1e-4
        return None                        # missed detection / below threshold
    quality_db  = 10.0 * np.log10(z_obs)

    # --- CRLB noise covariances (use realised xi_actual for this frame) ---
    crlb_vars = compute_crlb_variances(xi_actual, cfg, az_deg, el_deg)
    R_sim     = compute_r_sim(crlb_vars)   # 4×4 diagonal

    # --- sample noisy measurement from N(truth, R_sim) ---
    std_devs = np.sqrt(np.diag(R_sim))     # [σ_R, σ_vr, σ_az, σ_el]
    noise    = rng.standard_normal(4) * std_devs

    R_noisy   = R      + noise[0]
    vr_noisy  = vr     + noise[1]
    az_noisy  = az_deg + noise[2]
    el_noisy  = el_deg + noise[3]

    # --- beam-centre angles ---
    theta_b, phi_b = BEAM_CENTRES[beam_b]

    d_m = np.array([R_noisy, vr_noisy, az_noisy, el_noisy,
                    theta_b, phi_b, t_beam], dtype=np.float64)
    return d_m, quality_db


# ---------------------------------------------------------------------------
# 6.  Convenience: scan over all 32 beams for one target
# ---------------------------------------------------------------------------

def scan_target(
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    cfg: dict,
    sic_state_db: float,
    u_rcs: float,
    rng: np.random.Generator,
    cpi_per_beam_s: float | None = None,
) -> list[np.ndarray]:
    """Generate detections across the full 32-beam scan for one target.

    Beams are visited sequentially, each dwell lasting CPI = cfg['cpi_s'].
    The Swerling-I RCS draw u_rcs is constant across all 32 beams (one
    coherence interval = one full scan).

    Parameters
    ----------
    cpi_per_beam_s : override dwell time per beam (default: cfg['cpi_s'])

    Returns
    -------
    detections : list of (8,) ndarrays = [R̃, ṽ_r, θ̃_az, φ̃_el, θ_b, φ_b, δt_m, Q_m]
    """
    dwell = cpi_per_beam_s if cpi_per_beam_s is not None else cfg['cpi_s']
    detections = []
    for b in range(32):
        # δt_m < 0: time before the scan's reporting boundary (matches scene.py convention)
        t_beam = -(32 - b - 0.5) * dwell
        result = generate_detection(target_pos, target_vel, b, cfg,
                                    sic_state_db, u_rcs, rng, t_beam)
        if result is not None:
            d_m, quality_db = result
            detections.append(np.append(d_m, quality_db))
    return detections
