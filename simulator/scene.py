"""
simulator/scene.py
==================
Outer simulation loop for RCATT.

Produces one Frame dict per time step:
  {
    'k':          int,          # frame index
    'cfg_key':    str,          # e.g. 'C1'
    'sic_db':     float,        # SIC residual power, dB
    'context':    np.ndarray,   # (5,) c_k = [B_k, N_pil,k, T_obs,k, P_tx_k, P̂_SI,k]
    'states':     np.ndarray,   # (N, 6)  [x,y,z,vx,vy,vz] in m / m·s⁻¹
    'ids':        list[int],    # target IDs (length N)
    'modes':      np.ndarray,   # (N,)  motion mode: 0=CV,1=CA,2=CT
    'omegas':     np.ndarray,   # (N,)  turn rate rad/s (0 for CV/CA)
    'visible':    np.ndarray,   # (N,) bool — True if target inside surveillance volume
    'detections': np.ndarray,   # (M, 8)  per-detection feature vectors; ≤1 per target per scan
    'labels':     np.ndarray,   # (M,)  0=clutter, target_id for true detections
  }

Detection feature vector d_m ∈ ℝ⁸:
  [R̃, ṽ_r, θ̃_az, φ̃_el, θ_b, φ_b, δt_m, Q_m]

  δt_m = t_beam - t_k  (always ≤ 0, relative to the reporting boundary t_k)
  Range: −T_SCAN ≤ δt_m ≤ 0, where T_SCAN = N_BEAMS × CPI.
  Q_m  = 10 log₁₀(Z) dB — observed matched-filter output power,
         where Z is the matched-filter test statistic; the observable per-detection quality estimate.

Context token c_k ∈ ℝ⁵:
  [B_PRS,k (MHz), N_pil,k, T_obs,k (ms), P_tx,k (dBm), P̂_SI,k (dBm)]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config.sim_config import (
    # Surveillance volume
    SURV_X_RANGE, SURV_Y_RANGE, SURV_Z_RANGE,
    V_MAX_MPS, V_HORIZ_STD, V_VERT_STD,
    # Motion model
    CA_SIGMA_A, CT_OMEGA_MAX_DEGPS, CT_OMEGA_MIN_DEGPS,
    SIGMA_CV, SIGMA_CT,
    MOTION_TRANSITION,
    # Birth / death / clutter
    T_FRAME, LAMBDA_BIRTH, P_SURVIVAL, N_TARGET_MAX, LAMBDA_CLUTTER,
    # SIC
    SIC_STATES_DB, SIC_TRANSITION, SIC_POWER_EST_STD_DB,
    # Resource configs
    RESOURCE_CONFIGS,
    # Beam scan
    N_BEAMS,
    # RCS
    SIGMA_RCS_MEAN,
)
from simulator.beam_steering import BEAM_CENTRES as _BEAM_CENTRES_RAW, beam_gain_tx
from simulator.measurement_model import generate_detection, state_to_polar
from simulator.geometry import target_in_fov

# Convert once to numpy for vectorised indexing
BEAM_CENTRES_NP = np.array(_BEAM_CENTRES_RAW, dtype=float)  # (N_BEAMS, 2)  az, el [deg]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _birth_state(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a new target state [x, y, z, vx, vy, vz].

    Position sampled uniformly in Cartesian volume via rejection sampling so
    that ALL surveillance constraints are satisfied exactly:
      • slant range from gNB ≤ SURV_RHO_MAX (300 m)
      • azimuth (0° = North / +y, atan2(Δx, Δy)) ∈ [−60°, +60°]
      • elevation ∈ [10°, 40°]
      • UAV absolute altitude z ∈ [UAV_ALT_MIN, UAV_ALT_MAX] (50–150 m)

    Bounding box drawn from:
      x ∈ [−R_max·sin(60°), +R_max·sin(60°)]  ≈ [−260, +260] m
      y ∈ [0, R_max]                           = [0, 300] m  (az ∈ ±60° → y ≥ 0)
      z ∈ [UAV_ALT_MIN, UAV_ALT_MAX]           = [50, 150] m

    Velocity Gaussian with speed cap.
    """
    from config.sim_config import (
        SURV_RHO_MAX, GNB_POS,
        SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
        SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
        UAV_ALT_MIN, UAV_ALT_MAX,
    )

    x_half = SURV_RHO_MAX * np.sin(np.deg2rad(SURV_AZ_MAX_DEG))  # ≈ 259.8 m

    while True:
        x = rng.uniform(-x_half, x_half)
        y = rng.uniform(0.0, SURV_RHO_MAX)         # y ≥ 0 for az ∈ [−60°, 60°]
        z = rng.uniform(UAV_ALT_MIN, UAV_ALT_MAX)  # UAV altitude above ground

        # Vector from gNB to candidate position
        dx = x - GNB_POS[0]
        dy = y - GNB_POS[1]
        dz = z - GNB_POS[2]
        R  = np.sqrt(dx*dx + dy*dy + dz*dz)

        if R > SURV_RHO_MAX or R < 1e-6:
            continue

        # Azimuth convention: 0° = North (+y), consistent with beam_steering and birth
        az = float(np.degrees(np.arctan2(dx, dy)))
        el = float(np.degrees(np.arcsin(np.clip(dz / R, -1.0, 1.0))))

        if az < SURV_AZ_MIN_DEG or az > SURV_AZ_MAX_DEG:
            continue
        if el < SURV_EL_MIN_DEG or el > SURV_EL_MAX_DEG:
            continue

        break  # all constraints satisfied

    vx = rng.normal(0.0, V_HORIZ_STD)
    vy = rng.normal(0.0, V_HORIZ_STD)
    vz = rng.normal(0.0, V_VERT_STD)

    # Speed cap
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    if speed > V_MAX_MPS:
        scale = V_MAX_MPS / speed
        vx, vy, vz = vx * scale, vy * scale, vz * scale

    return np.array([x, y, z, vx, vy, vz], dtype=float)


def _motion_matrices_cv(dt: float):
    """CV: constant-velocity F and Q (6×6)."""
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Singer model: q = σ²_a * [dt³/3, dt²/2; dt²/2, dt] per axis
    s2 = SIGMA_CV ** 2
    q_block = np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ]) * s2
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[np.ix_([i, i + 3], [i, i + 3])] = q_block
    return F, Q


def _motion_matrices_ca(dt: float):
    """CA: same F and Q as CV (§15: CWNA, same spectral density).
    Persistent acceleration bias G·a_i is applied separately in _propagate."""
    return _motion_matrices_cv(dt)


def _ct_state_matrix(omega_rad: float, dt: float) -> np.ndarray:
    """
    Coordinated-turn F in horizontal plane (x-y), constant z velocity.
    State order: [x, y, z, vx, vy, vz].
    """
    if abs(omega_rad) < 1e-8:
        return _motion_matrices_cv(dt)[0]

    s  = np.sin(omega_rad * dt)
    c  = np.cos(omega_rad * dt)
    so = s / omega_rad
    co = (1 - c) / omega_rad

    F = np.zeros((6, 6))
    # x row
    F[0, 0] = 1.0;  F[0, 3] = so;   F[0, 4] = -co
    # y row
    F[1, 1] = 1.0;  F[1, 3] = co;   F[1, 4] = so
    # z row (constant velocity)
    F[2, 2] = 1.0;  F[2, 5] = dt
    # vx row
    F[3, 3] = c;    F[3, 4] = -s
    # vy row
    F[4, 3] = s;    F[4, 4] = c
    # vz row
    F[5, 5] = 1.0
    return F


def _ct_process_noise(omega_rad: float, dt: float) -> np.ndarray:
    """Process noise for CT mode (same variance as CA in horizontal plane)."""
    # Use SIGMA_CT for the noise level; apply it uniformly via Singer block
    s2 = SIGMA_CT ** 2
    q_block = np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ]) * s2
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[np.ix_([i, i + 3], [i, i + 3])] = q_block
    return Q


def _propagate_det(
    state: np.ndarray,
    mode: int,
    omega: float,
    accel: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Deterministic (noiseless) propagation over interval dt.
    Retained as a standalone utility for unit tests and analytical
    comparisons; the main generation loop uses _propagate() exclusively.
    Supports CV (linear), CA (quadratic), and CT (rotation matrix).
    """
    if mode == 0:          # CV
        F, _ = _motion_matrices_cv(dt)
        return F @ state
    elif mode == 1:        # CA — same F as CV plus deterministic acceleration
        F, _ = _motion_matrices_ca(dt)
        s = F @ state
        g_bias = np.empty(6)
        g_bias[:3] = 0.5 * dt * dt * accel
        g_bias[3:]  = dt * accel
        return s + g_bias
    else:                  # CT — exact rotation matrix
        F = _ct_state_matrix(omega, dt)
        return F @ state


def _propagate(
    state: np.ndarray,
    mode: int,
    omega: float,
    accel: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Propagate state vector one step under motion model *mode*.

    For CA (mode=1) a persistent acceleration bias a_i ∈ ℝ³ (held for the
    whole mode visit) is added via the input matrix G (§15):
      x^{k+1} = F_CV x^k + G a_i + w^k,  G = [T²/2·I₃; T·I₃]^T
    The noise w^k uses Q_CV (same CWNA spectral density as CV mode).
    """
    if mode == 0:          # CV
        F, Q = _motion_matrices_cv(dt)
    elif mode == 1:        # CA
        F, Q = _motion_matrices_ca(dt)   # = _motion_matrices_cv(dt)
    else:                  # CT
        F = _ct_state_matrix(omega, dt)
        Q = _ct_process_noise(omega, dt)

    noise = rng.multivariate_normal(np.zeros(6), Q)
    new_state = F @ state + noise

    if mode == 1:          # add deterministic acceleration bias G·a_i
        g_bias = np.empty(6)
        g_bias[:3] = 0.5 * dt * dt * accel   # T²/2 · a → position
        g_bias[3:]  = dt * accel              # T   · a → velocity
        new_state += g_bias

    # Speed cap
    speed = np.linalg.norm(new_state[3:6])
    if speed > V_MAX_MPS:
        new_state[3:6] *= V_MAX_MPS / speed

    return new_state


def _next_mode(mode: int, rng: np.random.Generator) -> tuple[int, float]:
    """
    Sample next motion mode from transition matrix.
    Returns (new_mode, omega_rad):
      omega_rad is 0 for CV/CA, randomly drawn for CT.
    """
    probs = MOTION_TRANSITION[mode]
    new_mode = rng.choice(3, p=probs)

    omega = 0.0
    if new_mode == 2:   # CT: draw turn rate magnitude uniformly, random sign
        omega_min = np.deg2rad(CT_OMEGA_MIN_DEGPS)
        omega_max = np.deg2rad(CT_OMEGA_MAX_DEGPS)
        mag   = rng.uniform(omega_min, omega_max)
        omega = mag * rng.choice([-1, 1])

    return new_mode, omega


def _next_sic_idx(idx: int, rng: np.random.Generator) -> int:
    """Sample next SIC state index from transition matrix."""
    probs = SIC_TRANSITION[idx]
    return int(rng.choice(len(SIC_STATES_DB), p=probs))


def make_context(cfg: dict, sic_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Build context token c_k ∈ ℝ⁵:
      [B_PRS,k (MHz), N_pil,k, T_obs,k (ms), P_tx,k (dBm), P̂_SI,k (dBm)]

    B_PRS,k = 12 * n_rb_prs * DELTA_F / 1e6  — occupied PRS bandwidth in MHz.
    P̂_SI,k  = P_tx,k - SIC_k + N(0, SIC_POWER_EST_STD_DB)  — residual SI power
               estimate in dBm (negative for strong SIC; e.g. 23 - 110 = -87 dBm).
    """
    from config.sim_config import DELTA_F
    b_prs_mhz    = 12 * cfg['n_rb_prs'] * DELTA_F / 1e6
    n_pil        = float(cfg['n_pil'])
    t_obs_ms     = cfg['t_obs_s'] * 1e3
    p_tx_dbm     = float(cfg['p_tx_dbm'])
    p_si_est_dbm = p_tx_dbm - sic_db + rng.normal(0.0, SIC_POWER_EST_STD_DB)

    return np.array([b_prs_mhz, n_pil, t_obs_ms, p_tx_dbm, p_si_est_dbm], dtype=float)


# ---------------------------------------------------------------------------
# Clutter generation
# ---------------------------------------------------------------------------

def _gen_clutter(n: int, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n clutter detection vectors, each in R^8.

    Positions are drawn by rejection sampling inside the true surveillance
    volume -- identical geometry to _birth_state (range <= SURV_RHO_MAX,
    az in [AZ_MIN, AZ_MAX], el in [EL_MIN, EL_MAX], altitude in
    [UAV_ALT_MIN, UAV_ALT_MAX]) -- so clutter is indistinguishable from
    real detections by spatial location alone.  The assigned beam is the
    highest-gain beam for the sampled direction; measurements add floor-level
    noise.

    The 8th element Q_m is the apparent SINR of a noise-only fixed-threshold
    crossing.  For a noise-only test statistic Z ~ Exp(1), the conditional
    distribution given a false alarm (Z > DETECTION_THRESHOLD) is
    DETECTION_THRESHOLD + Exp(1) by the memoryless property.
    Converted to dB this gives low Q_m values (≈ 10–13 dB for P_FA = 1e-4)
    regardless of the sampled position, clearly separating clutter from
    close-range true detections.

    Returns array of shape (n, 8); empty (0, 8) if n == 0.
    """
    if n == 0:
        return np.zeros((0, 8), dtype=float)

    from config.sim_config import (
        SURV_RHO_MAX, GNB_POS,
        SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
        SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
        UAV_ALT_MIN, UAV_ALT_MAX,
        FLOOR_RANGE_M, FLOOR_AZ_DEG, FLOOR_EL_DEG,
        DETECTION_THRESHOLD,
    )

    x_half = SURV_RHO_MAX * np.sin(np.deg2rad(SURV_AZ_MAX_DEG))

    rows = []
    while len(rows) < n:
        # Rejection-sample a position inside the surveillance volume,
        # mirroring _birth_state exactly (no velocity needed here).
        x = rng.uniform(-x_half, x_half)
        y = rng.uniform(0.0, SURV_RHO_MAX)
        z = rng.uniform(UAV_ALT_MIN, UAV_ALT_MAX)

        dx = x - GNB_POS[0]
        dy = y - GNB_POS[1]
        dz = z - GNB_POS[2]
        R  = np.sqrt(dx*dx + dy*dy + dz*dz)

        if R > SURV_RHO_MAX or R < 1e-6:
            continue

        az = float(np.degrees(np.arctan2(dx, dy)))
        el = float(np.degrees(np.arcsin(np.clip(dz / R, -1.0, 1.0))))

        if az < SURV_AZ_MIN_DEG or az > SURV_AZ_MAX_DEG:
            continue
        if el < SURV_EL_MIN_DEG or el > SURV_EL_MAX_DEG:
            continue

        # Position is inside the FOV -- assign beam and add measurement noise.
        gains    = np.array([beam_gain_tx(az, el, _BEAM_CENTRES_RAW, b)
                             for b in range(N_BEAMS)])
        beam_idx = int(np.argmax(gains))

        beam_az = BEAM_CENTRES_NP[beam_idx, 0]
        beam_el = BEAM_CENTRES_NP[beam_idx, 1]
        dt_m    = -(N_BEAMS - beam_idx - 0.5) * cfg['cpi_s']

        r_tilde  = max(1.0, R  + rng.normal(0.0, FLOOR_RANGE_M))
        az_tilde = az + rng.normal(0.0, FLOOR_AZ_DEG)
        el_tilde = el + rng.normal(0.0, FLOOR_EL_DEG)
        vr_tilde = rng.uniform(-V_MAX_MPS, V_MAX_MPS)

        # Quality feature: noise-only fixed-threshold crossing statistic.
        # Z ~ Exp(1); conditional on Z > DETECTION_THRESHOLD: Z = DETECTION_THRESHOLD + Exp(1).
        q_m = 10.0 * np.log10(DETECTION_THRESHOLD + rng.exponential(1.0))

        rows.append([r_tilde, vr_tilde, az_tilde, el_tilde, beam_az, beam_el, dt_m, q_m])

    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Main scene runner
# ---------------------------------------------------------------------------

def run_scene(
    cfg_keys: list[str],
    n_frames: int,
    rng: np.random.Generator,
    *,
    initial_sic_idx: Optional[int] = None,
    initial_n_targets: Optional[int] = None,
) -> list[dict]:
    """
    Simulate one complete sequence.

    Parameters
    ----------
    cfg_keys      : list of resource-config labels to draw from (e.g. TRAIN_CONFIGS)
    n_frames      : number of frames in the sequence
    rng           : seeded numpy Generator
    initial_sic_idx   : override initial SIC state index (0/1/2)
    initial_n_targets : override initial number of targets (≥0)

    Returns
    -------
    list of Frame dicts (length n_frames)
    """
    # ---- Initialise SIC state ----
    sic_idx = (
        int(rng.integers(0, len(SIC_STATES_DB)))
        if initial_sic_idx is None
        else initial_sic_idx
    )

    # ---- Initialise targets ----
    n_init = (
        int(rng.integers(1, N_TARGET_MAX + 1))
        if initial_n_targets is None
        else initial_n_targets
    )
    next_id = 1  # 0 reserved for clutter; UAV IDs start at 1

    states        = []  # list of (6,) arrays
    ids           = []  # list of int
    modes         = []  # list of int  (0=CV, 1=CA, 2=CT)
    omegas        = []  # list of float (turn rate rad/s)
    accels        = []  # list of (3,) arrays — persistent CA acceleration (§15)

    for _ in range(n_init):
        s = _birth_state(rng)
        m = int(rng.integers(0, 3))
        om = 0.0
        if m == 2:
            omega_min = np.deg2rad(CT_OMEGA_MIN_DEGPS)
            omega_max = np.deg2rad(CT_OMEGA_MAX_DEGPS)
            om = rng.uniform(omega_min, omega_max) * rng.choice([-1, 1])
        ac = rng.normal(0.0, CA_SIGMA_A, 3) if m == 1 else np.zeros(3)
        states.append(s)
        ids.append(next_id)
        modes.append(m)
        omegas.append(om)
        accels.append(ac)
        next_id += 1

    frames = []

    for k in range(n_frames):
        # --- Resource config for this frame ---
        cfg_key = cfg_keys[k % len(cfg_keys)]  # cycle or choose deterministically
        cfg     = RESOURCE_CONFIGS[cfg_key]

        # --- SIC state ---
        sic_db = float(SIC_STATES_DB[sic_idx])

        # --- Context token ---
        context = make_context(cfg, sic_db, rng)

        # --- Gather current states ---
        N = len(states)
        modes_arr  = np.array(modes,  dtype=int)   if N > 0 else np.zeros(0, dtype=int)
        omegas_arr = np.array(omegas, dtype=float) if N > 0 else np.zeros(0)

        T_SCAN = N_BEAMS * cfg['cpi_s']

        # Ground-truth states and visibility are assembled after the detection loop
        # from the stochastic trajectory endpoints (s_tk_list).
        # Invariant: s_truth(t_k) == s_next_frame_start(t_k) exactly.

        # --- Generate detections for each beam ---
        all_detections = []
        all_labels     = []

        # s_tk_list[n_idx] = stochastic state at reporting boundary t_k.
        # Built here; reused as ground truth AND as the next frame's starting state,
        # so s_truth(t_k) == s_next_frame_start(t_k) exactly.
        s_tk_list = []

        # Draw Swerling-I RCS multipliers once per target per frame
        u_rcs_all = rng.exponential(1.0, N) if N > 0 else np.zeros(0)

        # Best-beam selection: one detection per target per scan.
        # The 32-beam scan occupies the final T_SCAN = N_BEAMS * cpi_s seconds of
        # each 500 ms reporting frame.  One continuous stochastic trajectory is
        # propagated through all 32 beam midpoints to the reporting boundary t_k.
        for n_idx in range(N):
            s       = states[n_idx]
            mode_n  = modes[n_idx]
            omega_n = omegas[n_idx]
            accel_n = accels[n_idx]

            # One continuous stochastic trajectory: frame-start → 32 beam midpoints
            # → reporting boundary.  Sequential _propagate() calls accumulate
            # process noise correctly and produce s_tk for the invariant.
            cpi         = cfg['cpi_s']
            beam_states = []
            gain_list   = []

            # Frame-start → beam-0 midpoint
            t_first = T_FRAME - T_SCAN + 0.5 * cpi
            s_prev  = _propagate(s, mode_n, omega_n, accel_n, t_first, rng)
            _, az_b, el_b, _ = state_to_polar(s_prev[:3], s_prev[3:6])
            gain_list.append(beam_gain_tx(az_b, el_b, _BEAM_CENTRES_RAW, 0))
            beam_states.append(s_prev)

            # Beam b-1 midpoint → beam b midpoint  (dt = cpi_s each step)
            for b in range(1, N_BEAMS):
                s_next = _propagate(s_prev, mode_n, omega_n, accel_n, cpi, rng)
                _, az_b, el_b, _ = state_to_polar(s_next[:3], s_next[3:6])
                gain_list.append(beam_gain_tx(az_b, el_b, _BEAM_CENTRES_RAW, b))
                beam_states.append(s_next)
                s_prev = s_next

            # Beam-31 midpoint → reporting boundary t_k  (dt = 0.5 × cpi_s)
            s_tk = _propagate(s_prev, mode_n, omega_n, accel_n, 0.5 * cpi, rng)
            s_tk_list.append(s_tk)

            best_b    = int(np.argmax(gain_list))
            s_at_beam = beam_states[best_b]           # cached — no recomputation
            delta_t_m = -(N_BEAMS - best_b - 0.5) * cfg['cpi_s']

            det = generate_detection(
                target_pos   = s_at_beam[:3],
                target_vel   = s_at_beam[3:6],
                beam_b       = best_b,
                cfg          = cfg,
                sic_state_db = sic_db,
                u_rcs        = u_rcs_all[n_idx],
                rng          = rng,
                t_beam       = delta_t_m,
            )
            if det is not None:
                det_7d, quality_db = det
                all_detections.append(np.append(det_7d, quality_db))
                all_labels.append(ids[n_idx])

        # Assemble ground-truth states and visibility from stochastic endpoints
        if N > 0:
            states_tk = np.array(s_tk_list, dtype=float)
            visible   = np.array([target_in_fov(s) for s in states_tk], dtype=bool)
        else:
            states_tk = np.zeros((0, 6), dtype=float)
            visible   = np.zeros(0, dtype=bool)

        # Clutter: Poisson(λ_clutter) false alarms
        n_clutter = rng.poisson(LAMBDA_CLUTTER)
        if n_clutter > 0:
            clutter_dets = _gen_clutter(n_clutter, cfg, rng)
            for row in clutter_dets:
                all_detections.append(row)
                all_labels.append(0)   # 0 = clutter

        # Assemble detection arrays
        if all_detections:
            det_arr   = np.array(all_detections, dtype=float)
            label_arr = np.array(all_labels,     dtype=int)
            # Shuffle so position in array is not a discriminative feature
            perm      = rng.permutation(det_arr.shape[0])
            det_arr   = det_arr[perm]
            label_arr = label_arr[perm]
        else:
            det_arr   = np.zeros((0, 8), dtype=float)
            label_arr = np.zeros(0,      dtype=int)

        # --- Store frame ---
        frames.append({
            'k':          k,
            'cfg_key':    cfg_key,
            'sic_db':     sic_db,
            'context':    context,
            'states':     states_tk.copy(),
            'ids':        list(ids),
            'modes':      modes_arr.copy(),
            'omegas':     omegas_arr.copy(),
            'visible':    visible.copy(),
            'detections': det_arr,
            'labels':     label_arr,
        })

        # ---- State transition for next frame ----

        # 1. Propagate surviving targets
        new_states, new_ids, new_modes, new_omegas, new_accels = [], [], [], [], []
        for n_idx in range(N):
            if rng.uniform() < P_SURVIVAL:
                # Reuse stochastic endpoint from this frame's detection pass.
                # Invariant: s_truth(t_k) == s_next_frame_start(t_k) exactly.
                ns = s_tk_list[n_idx]
                nm, nom = _next_mode(modes[n_idx], rng)

                # Persistent acceleration for next frame (§15):
                #   entering CA → draw fresh a_i; staying in CA → hold; leaving → zero
                if nm == 1 and modes[n_idx] != 1:
                    nac = rng.normal(0.0, CA_SIGMA_A, 3)   # mode entry: draw
                elif nm == 1:
                    nac = accels[n_idx]                     # self-transition: hold
                else:
                    nac = np.zeros(3)                       # leaving CA: reset

                new_states.append(ns)
                new_ids.append(ids[n_idx])
                new_modes.append(nm)
                new_omegas.append(nom)
                new_accels.append(nac)

        # 2. Births — Poisson(λ_birth), capped at N_TARGET_MAX
        n_births = rng.poisson(LAMBDA_BIRTH)
        n_births_blocked = 0
        for _ in range(n_births):
            if len(new_states) >= N_TARGET_MAX:
                n_births_blocked += 1
                continue
            bs = _birth_state(rng)
            bm = int(rng.integers(0, 3))
            bom = 0.0
            if bm == 2:
                omega_min = np.deg2rad(CT_OMEGA_MIN_DEGPS)
                omega_max = np.deg2rad(CT_OMEGA_MAX_DEGPS)
                bom = rng.uniform(omega_min, omega_max) * rng.choice([-1, 1])
            bac = rng.normal(0.0, CA_SIGMA_A, 3) if bm == 1 else np.zeros(3)
            new_states.append(bs)
            new_ids.append(next_id)
            new_modes.append(bm)
            new_omegas.append(bom)
            new_accels.append(bac)
            next_id += 1

        # 3. Update SIC state
        sic_idx = _next_sic_idx(sic_idx, rng)

        states = new_states
        ids    = new_ids
        modes  = new_modes
        omegas = new_omegas
        accels = new_accels

    return frames
