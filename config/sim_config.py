"""
RCATT Simulator Configuration
==============================
Authoritative parameter set from RCATT_Measurement_Generation_Reference_v3.pdf.
All values verified against 3GPP TS 38.104 (Table 5.3.2-1) and TS 38.211.

Do NOT modify without updating manuscript Table III and re-running
acceptance tests G1-G9.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT      = 299_792_458.0       # m/s
K_BOLTZMANN  = 1.380_648_52e-23    # J/K

# ---------------------------------------------------------------------------
# 5G NR waveform (µ=1 numerology, Comb-2 DL-PRS)
# ---------------------------------------------------------------------------
F_C             = 3.5e9            # carrier frequency, Hz
LAMBDA          = C_LIGHT / F_C    # wavelength ≈ 85.7 mm
DELTA_F         = 30e3             # subcarrier spacing, Hz
SLOT_DURATION   = 0.5e-3           # slot duration, s  (1/2 ms for µ=1)
N_SYM_PER_SLOT  = 14               # OFDM symbols per slot
COMB_K          = 2                # Comb-2 PRS
# Pilots per RB per slot: (12 sc/RB) / Comb-2 × 4 PRS symbols/slot = 24
N_PIL_PER_RB_PER_SLOT = 24

# ---------------------------------------------------------------------------
# Receiver hardware
# ---------------------------------------------------------------------------
NOISE_FIGURE_DB  = 5.0
NOISE_FIGURE_LIN = 10 ** (NOISE_FIGURE_DB / 10)   # ≈ 3.162
T0               = 290.0                           # reference temp, K
N0               = K_BOLTZMANN * T0 * NOISE_FIGURE_LIN   # noise PSD, W/Hz

# ---------------------------------------------------------------------------
# UPA antenna array  (8×8 untilted vertical panel)
# ---------------------------------------------------------------------------
N_ANT_AZ       = 8
N_ANT_EL       = 8
N_ANT          = N_ANT_AZ * N_ANT_EL    # 64 elements
BORESIGHT_AZ_DEG = 0.0                  # panel boresight (unused; kept for reference)
BORESIGHT_EL_DEG = 0.0                  # untilted; electronic steering only
G_RX           = float(N_ANT)           # coherent Rx combining gain (linear) = 64

# ---------------------------------------------------------------------------
# Target / RCS model  (Swerling-I)
# ---------------------------------------------------------------------------
SIGMA_RCS_MEAN = 0.1    # mean RCS, m²  (−10 dBsm)
                        # draw: u ~ Exp(1), sigma = u * SIGMA_RCS_MEAN

# ---------------------------------------------------------------------------
# Scenario geometry
# ---------------------------------------------------------------------------
GNB_POS        = np.array([0.0, 0.0, 10.0])   # gNB, m  (UMi small-cell)
SURV_AZ_MIN_DEG =  -60.0
SURV_AZ_MAX_DEG =   60.0
SURV_EL_MIN_DEG =   10.0
SURV_EL_MAX_DEG =   40.0
SURV_RHO_MAX   = 300.0        # max slant range, m
UAV_ALT_MIN    =  50.0        # UAV altitude above ground, m
UAV_ALT_MAX    = 150.0        # (z_rel above gNB: 40–140 m)
N_BEAMS        = 32           # fixed pooled scan size per frame
V_SURV_M3      = 7.824e6      # surveillance volume, m³

# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
KAPPA          = 2.0     # inflation:  R_sim = kappa * R_CRLB + diag(floors)
FLOOR_RANGE_M  = 0.5     # range error floor, m
FLOOR_VEL_MPS  = 0.5     # radial-velocity error floor, m/s
FLOOR_AZ_DEG   = 0.5     # azimuth error floor, degrees
FLOOR_EL_DEG   = 0.5     # elevation error floor, degrees

# ---------------------------------------------------------------------------
# Detection threshold  (fixed threshold, matched-filter power test)
# ---------------------------------------------------------------------------
P_FA                = 1e-4             # target false-alarm rate
DETECTION_THRESHOLD = -np.log(P_FA)   # = 9.21034; PFA = exp(−DETECTION_THRESHOLD) = 1e-4
G_TX_MIN            = 10.0            # suppress detections with beam gain below this

# ---------------------------------------------------------------------------
# SIC Markov chain
# ---------------------------------------------------------------------------
SIC_STATES_DB   = np.array([100.0, 110.0, 120.0])   # isolation levels, dB
SIC_SELF_PROB   = 0.85
SIC_OFF_PROB    = 0.075
SIC_TRANSITION  = np.array([
    [SIC_SELF_PROB, SIC_OFF_PROB, SIC_OFF_PROB],
    [SIC_OFF_PROB,  SIC_SELF_PROB, SIC_OFF_PROB],
    [SIC_OFF_PROB,  SIC_OFF_PROB,  SIC_SELF_PROB],
])  # rows sum to 1: 0.85 + 2*0.075 = 1.0 ✓
SIC_POWER_EST_STD_DB = 1.5   # log-power estimation error std, dB

# ---------------------------------------------------------------------------
# Birth / death / clutter
# ---------------------------------------------------------------------------
T_FRAME        = 0.5      # frame duration, s
LAMBDA_BIRTH   = 0.1      # mean new targets per frame (Poisson)
P_SURVIVAL     = 0.95     # target survival probability per frame
# Rationale: P_SURVIVAL=0.95 gives mean geometric lifetime of 20 frames =
# full sequence duration; P(survive all 20 frames) ≈ 36%.  LAMBDA_BIRTH=0.1
# preserves E[simultaneous targets] = λ/(1−p) = 0.1/0.05 = 2.0 ≈ prior 2.21.
# With uniform initial draw N_init ∈ [1,5] (mean 3), scenes start dense and
# targets persist, giving the transformer sustained maneuvering tracks to learn
# from.  Prior 0.774/0.5 gave median lifetime 3 frames with 50% of tracks
# lasting only 1–2 frames — insufficient for temporal tracking demonstration.
N_TARGET_MAX   = 5
# Maximum number of physically active simulated UAVs,
# including targets outside the surveillance volume.
# This may temporarily suppress births when out-of-FOV
# targets occupy available slots.
#
# Future option: separate N_TARGET_MAX_VISIBLE from
# N_TARGET_MAX_TOTAL if this materially affects scene density.
LAMBDA_CLUTTER = 10.0     # mean clutter points per frame (Poisson)

# ---------------------------------------------------------------------------
# Jump-Markov motion model  (CV / CA / CT)
# ---------------------------------------------------------------------------
MODE_STAY_PROB = 0.75
MODE_OFF_PROB  = 0.125
MOTION_TRANSITION = np.array([
    [MODE_STAY_PROB, MODE_OFF_PROB, MODE_OFF_PROB],
    [MODE_OFF_PROB,  MODE_STAY_PROB, MODE_OFF_PROB],
    [MODE_OFF_PROB,  MODE_OFF_PROB,  MODE_STAY_PROB],
])  # modes: 0=CV, 1=CA, 2=CT
CA_SIGMA_A           = 2.0    # CA acceleration std, m/s²
CT_OMEGA_MAX_DEGPS   = 20.0   # CT max turn rate, °/s
CT_OMEGA_MIN_DEGPS   = 3.0    # CT minimum |turn rate|, °/s
SIGMA_CV             = 0.5    # CV process noise std, m/s²
SIGMA_CT             = 1.0    # CT process noise std, m/s²

# ---------------------------------------------------------------------------
# Surveillance volume (Cartesian, m; gNB at [0, 0, 10])
# ---------------------------------------------------------------------------
SURV_X_RANGE   = (-350.0, 350.0)
SURV_Y_RANGE   = (-350.0, 350.0)
SURV_Z_RANGE   = (30.0,   200.0)
V_MAX_MPS      = 25.0     # target speed cap, m/s
V_HORIZ_STD    = 10.0     # std of horizontal velocity at birth, m/s
V_VERT_STD     = 2.0      # std of vertical velocity at birth, m/s

# ---------------------------------------------------------------------------
# Dataset sizes
# ---------------------------------------------------------------------------
N_TRAIN            = 10_000
N_VAL              = 500
N_TEST             = 1_000
N_OOD              = 1_000
N_FRAMES_PER_SEQ   = 20       # frames per sequence (10 s at T_FRAME = 0.5 s)

# ---------------------------------------------------------------------------
# Resource configurations C1–C8
# ---------------------------------------------------------------------------
def compute_t_obs(n_slot: int) -> float:
    """First-to-last pilot time span (s).
    t_obs = ((N_slot-1)*14 + 3) * slot_duration/14
    C7 (n_slot=4)  → 1.607 ms
    C1 (n_slot=8)  → 3.607 ms
    C3 (n_slot=16) → 7.607 ms
    """
    return ((n_slot - 1) * N_SYM_PER_SLOT + 3) * (SLOT_DURATION / N_SYM_PER_SLOT)


def _cfg(n_rb_prs, n_sc_carrier, n_slot, p_tx_dbm, split):
    """Build one resource-config dict; all derived quantities computed here."""
    return dict(
        n_rb_prs     = n_rb_prs,
        n_sc_carrier = n_sc_carrier,          # carrier subcarriers (reference)
        n_pil        = N_PIL_PER_RB_PER_SLOT * n_rb_prs * n_slot,
        n_sym        = N_SYM_PER_SLOT * n_slot,
        n_slot       = n_slot,
        cpi_s        = SLOT_DURATION * n_slot,
        t_obs_s      = compute_t_obs(n_slot),
        p_tx_dbm     = p_tx_dbm,
        p_tx_w       = 10 ** ((p_tx_dbm - 30) / 10),
        split        = split,
    )

#              n_rb_prs  n_sc_carrier  n_slot  p_tx_dbm  split
# CPI (µ=1, 0.5 ms/slot): n_slot=4→2 ms, n_slot=8→4 ms, n_slot=16→8 ms
RESOURCE_CONFIGS = {
    'C1': _cfg( 48,   612,  8,  23, 'train'),  # n_pil= 9,216  CPI=4 ms
    'C2': _cfg(104,  1272,  8,  23, 'train'),  # n_pil=19,968  CPI=4 ms
    'C3': _cfg( 48,   612, 16,  26, 'train'),  # n_pil=18,432  CPI=8 ms
    'C4': _cfg(104,  1272, 16,  26, 'train'),  # n_pil=39,936  CPI=8 ms
    'C5': _cfg( 24,   288,  8,  20, 'ood'),    # n_pil= 4,608  OOD: narrow BW, CPI=4 ms
    'C6': _cfg(272,  3276,  8,  23, 'train'),  # n_pil=52,224  CPI=4 ms
    'C7': _cfg( 48,   612,  4,  30, 'ood'),    # n_pil= 4,608  OOD: short CPI, CPI=2 ms
    'C8': _cfg(132,  1596, 16,  29, 'train'),  # n_pil=50,688  CPI=8 ms
}

TRAIN_CONFIGS = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'train']
OOD_CONFIGS   = [k for k, v in RESOURCE_CONFIGS.items() if v['split'] == 'ood']
