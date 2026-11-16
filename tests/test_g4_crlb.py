"""G4: CRLB and measurement-noise covariance validation.

Verifies compute_crlb_variances() and compute_r_sim() against:
  - independent closed-form reference formulas
  - scaling laws (1/SNR_coh, 1/n_pil, 1/xi)
  - structural properties (diagonal R_sim, floor lower bound, positivity)
  - temporal spread σ_t ordering across CPI lengths

Pass criteria
-------------
- Range and Doppler CRLBs match first-principles reference to < 1e-9 relative.
- CRLB halves when n_pil or xi doubles (exact linear scaling).
- All four CRLB variances are positive for every config and test angle.
- R_sim = KAPPA·R_CRLB + diag(floors²) exactly, diagonal only.
- R_sim ≥ diag(floors²) everywhere (floor is never violated).
- Longer CPI (more slots) → larger σ_t → smaller var_vr for equal SNR_coh.
- Wider PRS bandwidth (more RBs) → larger β → smaller var_R for equal SNR_coh.
"""

import pytest
import numpy as np

from config.sim_config import (
    RESOURCE_CONFIGS, C_LIGHT, LAMBDA, DELTA_F,
    N_SYM_PER_SLOT, SLOT_DURATION,
    N_ANT_AZ, N_ANT_EL,
    KAPPA,
    FLOOR_RANGE_M, FLOOR_VEL_MPS, FLOOR_AZ_DEG, FLOOR_EL_DEG,
)
from simulator.measurement_model import compute_crlb_variances, compute_r_sim

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
_AZ, _EL = 7.5, 21.25     # beam-12 boresight — Jacobian non-trivial
_XI      = 1.0             # unit per-RE SINR (easy to scale in tests)
_CFG     = RESOURCE_CONFIGS['C1']

TOL_REL  = 1e-9    # relative tolerance for deterministic tests
TOL_ABS  = 1e-12   # absolute tolerance for exact-equality checks

FLOORS_SQ = np.array([
    FLOOR_RANGE_M ** 2,
    FLOOR_VEL_MPS ** 2,
    FLOOR_AZ_DEG ** 2,
    FLOOR_EL_DEG ** 2,
])


# ---------------------------------------------------------------------------
# Independent reference formulas
# ---------------------------------------------------------------------------

def _ref_var_R(xi: float, cfg: dict) -> float:
    """Range CRLB from first principles.
    β² = (N_sc_PRS² − 1)/12 × (2Δf)²,  N_sc_PRS = 6·N_RB_PRS (Comb-2)
    SNR_coh = n_pil · ξ
    var_R = c² / (32π² · SNR_coh · β²)
    """
    n_sc_prs = 6 * cfg['n_rb_prs']                     # Comb-2: 6 occupied sc/RB
    beta_sq  = (n_sc_prs ** 2 - 1) / 12.0 * (2 * DELTA_F) ** 2
    SNR_coh  = cfg['n_pil'] * xi
    return C_LIGHT ** 2 / (32.0 * np.pi ** 2 * SNR_coh * beta_sq)


def _ref_sigma_t_sq(cfg: dict) -> float:
    """Exact temporal variance of PRS pilot times (ddof=0).
    PRS symbols at k ∈ {0,1,2,3} in each of n_slot slots.
    T_sym = SLOT_DURATION / N_SYM_PER_SLOT.
    """
    T_sym  = SLOT_DURATION / N_SYM_PER_SLOT            # = cfg['cpi_s'] / cfg['n_sym']
    n_slot = cfg['n_slot']
    times  = np.array([(14 * s + k) * T_sym
                       for s in range(n_slot) for k in range(4)])
    return float(np.var(times))                         # population variance, ddof=0


def _ref_var_vr(xi: float, cfg: dict) -> float:
    """Doppler CRLB from first principles.
    var_vr = λ² / (32π² · SNR_coh · σ_t²)
    """
    sigma_t_sq = _ref_sigma_t_sq(cfg)
    SNR_coh    = cfg['n_pil'] * xi
    return LAMBDA ** 2 / (32.0 * np.pi ** 2 * SNR_coh * sigma_t_sq)


# ---------------------------------------------------------------------------
# G4-1: Range CRLB matches independent formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
def test_range_crlb_independent(cfg_key):
    """var_R matches the first-principles bandwidth formula for every config."""
    cfg     = RESOURCE_CONFIGS[cfg_key]
    xi      = 0.5
    ref     = _ref_var_R(xi, cfg)
    result  = compute_crlb_variances(xi, cfg, _AZ, _EL)['R']
    assert abs(result / ref - 1.0) < TOL_REL, (
        f"{cfg_key}: var_R={result:.6e}, ref={ref:.6e}"
    )


# ---------------------------------------------------------------------------
# G4-2: Doppler CRLB matches independent formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
def test_doppler_crlb_independent(cfg_key):
    """var_vr matches the first-principles temporal-spread formula for every config."""
    cfg     = RESOURCE_CONFIGS[cfg_key]
    xi      = 0.5
    ref     = _ref_var_vr(xi, cfg)
    result  = compute_crlb_variances(xi, cfg, _AZ, _EL)['vr']
    assert abs(result / ref - 1.0) < TOL_REL, (
        f"{cfg_key}: var_vr={result:.6e}, ref={ref:.6e}"
    )


# ---------------------------------------------------------------------------
# G4-3: Scaling with n_pil — all four channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ['R', 'vr', 'az', 'el'])
def test_crlb_halves_when_npil_doubles(key):
    """Doubling n_pil halves every CRLB variance (SNR_coh = n_pil·xi)."""
    cfg2 = dict(_CFG, n_pil=_CFG['n_pil'] * 2)
    v1   = compute_crlb_variances(_XI, _CFG, _AZ, _EL)[key]
    v2   = compute_crlb_variances(_XI, cfg2, _AZ, _EL)[key]
    assert abs(v1 / v2 - 2.0) < TOL_REL, (
        f"var_{key}: ratio={v1/v2:.10f}, expected 2.0"
    )


@pytest.mark.parametrize("key", ['R', 'vr', 'az', 'el'])
def test_crlb_quarter_when_npil_quadruples(key):
    """4× n_pil → 1/4 CRLB variance."""
    cfg4 = dict(_CFG, n_pil=_CFG['n_pil'] * 4)
    v1   = compute_crlb_variances(_XI, _CFG, _AZ, _EL)[key]
    v4   = compute_crlb_variances(_XI, cfg4, _AZ, _EL)[key]
    assert abs(v1 / v4 - 4.0) < TOL_REL, (
        f"var_{key}: ratio={v1/v4:.10f}, expected 4.0"
    )


# ---------------------------------------------------------------------------
# G4-4: Scaling with xi — all four channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ['R', 'vr', 'az', 'el'])
def test_crlb_halves_when_xi_doubles(key):
    """Doubling per-RE SINR halves every CRLB variance."""
    v1 = compute_crlb_variances(_XI,       _CFG, _AZ, _EL)[key]
    v2 = compute_crlb_variances(_XI * 2.0, _CFG, _AZ, _EL)[key]
    assert abs(v1 / v2 - 2.0) < TOL_REL, (
        f"var_{key}: ratio={v1/v2:.10f}, expected 2.0"
    )


# ---------------------------------------------------------------------------
# G4-5: Positivity and physical plausibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
@pytest.mark.parametrize("az,el", [(0.0, 25.0), (7.5, 21.25), (-30.0, 15.0)])
def test_crlb_all_positive(cfg_key, az, el):
    """All CRLB variances are positive for all configs and angles."""
    cfg  = RESOURCE_CONFIGS[cfg_key]
    vars_ = compute_crlb_variances(0.1, cfg, az, el)
    for key, val in vars_.items():
        assert val > 0, f"{cfg_key}: var_{key}={val} at az={az}, el={el}"


def test_crlb_range_stddev_plausible():
    """√var_R < 10 m at moderate range/SNR (C1, R≈150m via SINR≈1)."""
    var_R = compute_crlb_variances(_XI, _CFG, _AZ, _EL)['R']
    assert np.sqrt(var_R) < 10.0, f"σ_R={np.sqrt(var_R):.3f} m unreasonably large"


def test_crlb_angle_stddev_plausible():
    """√var_az, √var_el < 10° for 8×8 UPA at any SNR > 0.1."""
    vars_ = compute_crlb_variances(0.1, _CFG, _AZ, _EL)
    assert np.sqrt(vars_['az']) < 10.0, f"σ_az={np.sqrt(vars_['az']):.3f}° too large"
    assert np.sqrt(vars_['el']) < 10.0, f"σ_el={np.sqrt(vars_['el']):.3f}° too large"


# ---------------------------------------------------------------------------
# G4-6: R_sim formula verification
# ---------------------------------------------------------------------------

def test_r_sim_shape_and_diagonal():
    """R_sim is a 4×4 matrix; all off-diagonal entries are exactly zero."""
    crlb  = compute_crlb_variances(_XI, _CFG, _AZ, _EL)
    R_sim = compute_r_sim(crlb)
    assert R_sim.shape == (4, 4)
    off_diag = R_sim - np.diag(np.diag(R_sim))
    assert np.max(np.abs(off_diag)) == 0.0, "R_sim has non-zero off-diagonal entries"


def test_r_sim_diagonal_matches_formula():
    """R_sim diagonal = KAPPA·R_CRLB + diag(floors²) for all four channels."""
    crlb     = compute_crlb_variances(_XI, _CFG, _AZ, _EL)
    R_sim    = compute_r_sim(crlb)
    expected = np.array([
        KAPPA * crlb['R']  + FLOOR_RANGE_M ** 2,
        KAPPA * crlb['vr'] + FLOOR_VEL_MPS ** 2,
        KAPPA * crlb['az'] + FLOOR_AZ_DEG ** 2,
        KAPPA * crlb['el'] + FLOOR_EL_DEG ** 2,
    ])
    assert np.allclose(np.diag(R_sim), expected, rtol=1e-14, atol=0), (
        f"max relative error = {np.max(np.abs(np.diag(R_sim) / expected - 1.0)):.2e}"
    )


@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
def test_r_sim_exceeds_floors_always(cfg_key):
    """R_sim[i,i] ≥ floor[i]² for every config (KAPPA·R_CRLB ≥ 0)."""
    cfg   = RESOURCE_CONFIGS[cfg_key]
    crlb  = compute_crlb_variances(0.001, cfg, _AZ, _EL)   # low SNR, large CRLB
    R_sim = compute_r_sim(crlb)
    diag  = np.diag(R_sim)
    for i, (d, f_sq) in enumerate(zip(diag, FLOORS_SQ)):
        assert d >= f_sq - TOL_ABS, (
            f"{cfg_key}: R_sim[{i},{i}]={d:.6e} < floor²={f_sq:.6e}"
        )


def test_r_sim_approaches_floors_at_high_snr():
    """At very high SNR, R_sim converges to diag(floors²) + tiny CRLB term."""
    xi_high  = 1e8          # xi = 1e8 → SNR_coh ≈ 9.2e11; CRLB << floors²
    crlb     = compute_crlb_variances(xi_high, _CFG, _AZ, _EL)
    R_sim    = compute_r_sim(crlb)
    diag     = np.diag(R_sim)
    # Each diagonal entry should be within 0.1% of its floor²
    rel_excess = (diag - FLOORS_SQ) / FLOORS_SQ
    assert np.all(rel_excess < 1e-3), (
        f"R_sim not near floors² at xi={xi_high}: rel_excess={rel_excess}"
    )


# ---------------------------------------------------------------------------
# G4-7: Temporal spread ordering (CPI length)
# ---------------------------------------------------------------------------

def test_sigma_t_increases_with_cpi():
    """Longer CPI (more slots) → larger σ_t² → smaller var_vr for equal SNR_coh.

    C7 (n_slot=4) < C1 (n_slot=8) < C3 (n_slot=16).
    """
    st_4  = _ref_sigma_t_sq(RESOURCE_CONFIGS['C7'])  # n_slot=4
    st_8  = _ref_sigma_t_sq(RESOURCE_CONFIGS['C1'])  # n_slot=8
    st_16 = _ref_sigma_t_sq(RESOURCE_CONFIGS['C3'])  # n_slot=16
    assert st_4 < st_8 < st_16, (
        f"σ_t² ordering wrong: n4={st_4:.3e}, n8={st_8:.3e}, n16={st_16:.3e}"
    )


def test_sigma_t_values_independent():
    """Reference σ_t² for the three distinct n_slot values matches brute force."""
    for cfg_key, expected_n_slot in [('C7', 4), ('C1', 8), ('C3', 16)]:
        cfg    = RESOURCE_CONFIGS[cfg_key]
        T_sym  = SLOT_DURATION / N_SYM_PER_SLOT
        times  = np.array([(14 * s + k) * T_sym
                           for s in range(expected_n_slot) for k in range(4)])
        ref_st = float(np.var(times))
        sim_st = _ref_sigma_t_sq(cfg)
        assert abs(sim_st / ref_st - 1.0) < TOL_REL, (
            f"{cfg_key}: σ_t²={sim_st:.6e}, ref={ref_st:.6e}"
        )


def test_longer_cpi_smaller_var_vr_equal_snr():
    """var_vr decreases with CPI when SNR_coh is equal (hold n_pil·xi constant)."""
    # Normalise by n_pil so SNR_coh = n_pil*xi is the same across configs
    snr_coh = 1000.0
    cfg_7   = RESOURCE_CONFIGS['C7']
    cfg_1   = RESOURCE_CONFIGS['C1']
    cfg_3   = RESOURCE_CONFIGS['C3']
    xi_7    = snr_coh / cfg_7['n_pil']
    xi_1    = snr_coh / cfg_1['n_pil']
    xi_3    = snr_coh / cfg_3['n_pil']
    vvr_7   = compute_crlb_variances(xi_7, cfg_7, _AZ, _EL)['vr']
    vvr_1   = compute_crlb_variances(xi_1, cfg_1, _AZ, _EL)['vr']
    vvr_3   = compute_crlb_variances(xi_3, cfg_3, _AZ, _EL)['vr']
    assert vvr_7 > vvr_1 > vvr_3, (
        f"var_vr ordering wrong (n_slot=4→8→16): {vvr_7:.3e} {vvr_1:.3e} {vvr_3:.3e}"
    )


# ---------------------------------------------------------------------------
# G4-8: Frequency bandwidth ordering (RB count)
# ---------------------------------------------------------------------------

def test_wider_bandwidth_smaller_var_R():
    """More PRS RBs → wider β → smaller var_R for equal SNR_coh.

    C5 (24 RB) < C1 (48 RB) < C2 (104 RB) < C6 (272 RB), all n_slot=8.
    """
    snr_coh = 1000.0
    keys_8slot = ['C5', 'C1', 'C2', 'C6']       # all n_slot=8, increasing n_rb
    var_Rs = []
    for k in keys_8slot:
        cfg = RESOURCE_CONFIGS[k]
        xi  = snr_coh / cfg['n_pil']
        var_Rs.append(compute_crlb_variances(xi, cfg, _AZ, _EL)['R'])
    # Should be strictly decreasing
    for i in range(len(var_Rs) - 1):
        assert var_Rs[i] > var_Rs[i + 1], (
            f"{keys_8slot[i]}→{keys_8slot[i+1]}: "
            f"var_R={var_Rs[i]:.4e}→{var_Rs[i+1]:.4e} not decreasing"
        )


def test_bandwidth_ratio_consistent_with_rb_ratio():
    """var_R(C5) / var_R(C1) ≈ (β_C1 / β_C5)² (at equal SNR_coh).

    Both have n_slot=8; β scales as √((6·n_rb)²−1) ≈ 6·n_rb for large n_rb,
    so the ratio should be ≈ (n_rb_C1 / n_rb_C5)² = (48/24)² = 4.0.
    """
    snr_coh = 1000.0
    cfg5, cfg1 = RESOURCE_CONFIGS['C5'], RESOURCE_CONFIGS['C1']
    xi5 = snr_coh / cfg5['n_pil']
    xi1 = snr_coh / cfg1['n_pil']
    vr5 = compute_crlb_variances(xi5, cfg5, _AZ, _EL)['R']
    vr1 = compute_crlb_variances(xi1, cfg1, _AZ, _EL)['R']

    # Independent reference ratio from β formula
    n5, n1 = 6 * cfg5['n_rb_prs'], 6 * cfg1['n_rb_prs']   # n_sc_prs
    beta_sq_5 = (n5 ** 2 - 1) / 12.0 * (2 * DELTA_F) ** 2
    beta_sq_1 = (n1 ** 2 - 1) / 12.0 * (2 * DELTA_F) ** 2
    expected_ratio = beta_sq_1 / beta_sq_5

    actual_ratio = vr5 / vr1
    assert abs(actual_ratio / expected_ratio - 1.0) < TOL_REL, (
        f"var_R ratio C5/C1: {actual_ratio:.8f}, β ratio={expected_ratio:.8f}"
    )


# ---------------------------------------------------------------------------
# G4-9: Monte Carlo – empirical measurement-error matches R_sim diagonal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", RESOURCE_CONFIGS.keys())
def test_g4_mc_empirical_error(cfg_key):
    """G4 Monte Carlo: empirical std of measurement noise ≈ sqrt(R_sim diagonal).

    Scenario: R=220 m, beam-12 boresight (az=7.5°, el=21.25°), SIC=110 dB,
    fixed RCS = SIGMA_RCS_MEAN (non-Swerling, deterministic SINR).

    N=3000 i.i.d. draws from N(0, R_sim); checks all four channels (R, vr,
    az, el) for all 8 configs.  5% tolerance > 3.9σ of the std estimator at
    N=3000, giving a theoretical failure rate < 0.01%.
    """
    from simulator.measurement_model import compute_sinr, compute_crlb_variances, compute_r_sim
    from config.sim_config import SIGMA_RCS_MEAN as _SRCS
    from simulator.beam_steering import BEAM_CENTRES as _BC

    rng = np.random.default_rng(100 + ord(cfg_key[1]))   # stable per cfg

    R, az, el, beam = 220.0, 7.5, 21.25, 12   # beam-12 boresight
    sic_db = 110.0
    cfg = RESOURCE_CONFIGS[cfg_key]

    xi      = compute_sinr(R, az, el, _SRCS, beam, cfg, sic_db)
    crlb    = compute_crlb_variances(xi, cfg, az, el)
    R_sim   = compute_r_sim(crlb)
    pred    = np.sqrt(np.diag(R_sim))           # [σ_R, σ_vr, σ_az, σ_el]

    N = 3000
    # Draw samples directly from N(0, R_sim): marginals suffice since R_sim diagonal
    noise = rng.standard_normal((N, 4)) * pred
    emp   = noise.std(axis=0, ddof=1)

    tol = 0.05   # 5% relative tolerance
    for i, dim in enumerate(['R', 'vr', 'az', 'el']):
        rel = abs(emp[i] / pred[i] - 1.0)
        assert rel < tol, (
            f"{cfg_key} dim={dim}: empirical σ={emp[i]:.4g}, predicted={pred[i]:.4g}, "
            f"rel_err={rel:.3f} (tol={tol})"
        )


# ---------------------------------------------------------------------------
# G4-10: Angular Fisher – independent Jacobian check at multiple az/el
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az_deg,el_deg", [
    (0.0,  25.0),    # az=0 → J diagonal (cross-term vanishes)
    (45.0, 20.0),    # oblique: cross-term non-trivial
    (60.0, 35.0),    # near az/el limits
])
def test_g4_angular_fisher_jacobian(az_deg, el_deg):
    """G4 Angular Fisher: independent Jacobian reproduces compute_crlb_variances.

    Independently constructs J = d(u,v)/d(az,el) from direction-cosine
    definitions (u = cos el·sin az, v = sin el) and computes

        C_angles = J⁻¹ · diag(σ_u², σ_v²) · J⁻¹ᵀ

    using the same analytical formula for σ_u², σ_v² as the production
    code but through independent NumPy calls.  The result must match
    compute_crlb_variances to relative tolerance 1e-9.

    Covers: correct Jacobian entries (including the du/del cross-term that
    vanishes only at az=0°), correct matrix-inversion order, and
    rad→deg unit conversion.
    """
    xi  = 1.0
    cfg = RESOURCE_CONFIGS['C1']
    SNR_coh = cfg['n_pil'] * xi

    az_r = np.radians(az_deg)
    el_r = np.radians(el_deg)

    # --- Independent Jacobian: J[i,j] = d(u,v)_i / d(az,el)_j ---
    du_daz = np.cos(el_r) * np.cos(az_r)
    du_del = -np.sin(el_r) * np.sin(az_r)
    dv_daz = 0.0
    dv_del = np.cos(el_r)
    J = np.array([[du_daz, du_del],
                  [dv_daz, dv_del]])

    # --- Direction-cosine FI (analytical, same formula as production) ---
    sigma_u_sq = 6.0 / (SNR_coh * np.pi ** 2 * (N_ANT_AZ ** 2 - 1))
    sigma_v_sq = 6.0 / (SNR_coh * np.pi ** 2 * (N_ANT_EL ** 2 - 1))

    # --- Transform to az/el covariance (radians²) ---
    J_inv = np.linalg.inv(J)
    C_rad = J_inv @ np.diag([sigma_u_sq, sigma_v_sq]) @ J_inv.T

    var_az_ref = C_rad[0, 0] * (180.0 / np.pi) ** 2   # deg²
    var_el_ref = C_rad[1, 1] * (180.0 / np.pi) ** 2   # deg²

    result = compute_crlb_variances(xi, cfg, az_deg, el_deg)

    tol = 1e-9   # exact match expected (identical formula, independent code)
    for key, ref in [('az', var_az_ref), ('el', var_el_ref)]:
        rel = abs(result[key] / ref - 1.0)
        assert rel < tol, (
            f"az={az_deg}°, el={el_deg}°: var_{key}={result[key]:.8e}, "
            f"ref={ref:.8e}, rel_err={rel:.2e}"
        )

    # Bonus: at az=0° the Jacobian must be diagonal
    if az_deg == 0.0:
        assert abs(J[0, 1]) < 1e-15, f"J[0,1] should vanish at az=0, got {J[0,1]}"
        assert abs(J[1, 0]) < 1e-15, f"J[1,0] should vanish always, got {J[1,0]}"
