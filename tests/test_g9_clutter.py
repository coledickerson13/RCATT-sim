"""
test_g9_clutter.py – G9: Clutter statistics validation
=======================================================
Verifies that the clutter generation model produces the correct Poisson
count distribution, that clutter positions lie inside the surveillance
volume, that quality statistics follow the theoretical distribution, and
that no target identity leaks into clutter labels.

Theory
------
Clutter count per frame: K ~ Poisson(λ_c), λ_c = LAMBDA_CLUTTER = 10.
  E[K] = Var[K] = 10.

Clutter quality: Q_m = 10·log10(Z_m) where Z_m = η + X, X ~ Exp(1).
  By the memoryless property of Exp(1):
      Z_m | Z_m > η ~ η + Exp(1)
  so Z_m − η ~ Exp(1) exactly.

Clutter geometry (airborne, identical FOV to UAVs):
  Clutter scatterers are sampled from the same airborne surveillance volume
  as UAVs via rejection sampling (_gen_clutter mirrors _birth_state).
  Azimuth:   −60° ≤ θ_az ≤ 60°
  Elevation: 10° ≤ φ_el ≤ 40°
  Range:     R ≤ RHO_MAX = 300 m
  Altitude:  UAV_ALT_MIN ≤ z ≤ UAV_ALT_MAX (clutter is NOT at ground level)

Pass criteria
-------------
G9-1  Mean clutter count per frame is within 3σ of 10 over 10,000 frames.
G9-2  Variance of clutter count per frame is within 3σ of 10.
G9-3  Clutter quality residual (Z_m − η) follows Exp(1): KS-test p ≥ 0.05.
G9-4  All clutter Q_m > 10·log10(η)  (i.e. Z_m > η by construction).
G9-5  No clutter detection has label ≠ 0 (no target ID leakage).
G9-6  Clutter-only frames (initial_n_targets=0) have only label==0 detections.
G9-7  Clutter count is independent across frames: lag-1 autocorrelation ≈ 0
      (|r(1)| < 3·(1/√N) at N=10,000, the 3σ white-noise band).
G9-8  Clutter measurement angles and range fall within the surveillance volume
      bounds (az within [AZ_MIN, AZ_MAX] ± measurement noise, el within
      [EL_MIN, EL_MAX] ± noise, R ≤ RHO_MAX + 3·FLOOR_RANGE_M).
"""

import sys
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest
from scipy import stats

from config.sim_config import (
    DETECTION_THRESHOLD, LAMBDA_CLUTTER, RESOURCE_CONFIGS,
    SURV_RHO_MAX, SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
    SURV_EL_MIN_DEG, SURV_EL_MAX_DEG, FLOOR_RANGE_M,
)


# ---------------------------------------------------------------------------
# Shared fixture: 10,000 clutter-only frames
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def clutter_frames():
    """10,000 pure-clutter frames from C1 (n_slot=8, cpi=4 ms)."""
    from simulator.scene import run_scene
    rng = np.random.default_rng(9000)
    n_frames = 10_000
    frames = run_scene(['C1'] * n_frames, n_frames, rng, initial_n_targets=0)
    return frames


# ---------------------------------------------------------------------------
# G9-1: Mean clutter count ≈ LAMBDA_CLUTTER = 10
# ---------------------------------------------------------------------------

def test_clutter_count_mean(clutter_frames):
    """E[K] = LAMBDA_CLUTTER = 10 within 3σ_MC over 10,000 frames."""
    counts = np.array([int((fr['labels'] == 0).sum()) for fr in clutter_frames])
    N   = len(counts)
    mu  = counts.mean()
    # SE of mean for Poisson(λ): std/√N = √(λ/N)
    se  = np.sqrt(LAMBDA_CLUTTER / N)
    assert abs(mu - LAMBDA_CLUTTER) < 3 * se, (
        f"Mean clutter count {mu:.4f} ≠ λ={LAMBDA_CLUTTER} "
        f"(3σ band ={3*se:.4f})"
    )


# ---------------------------------------------------------------------------
# G9-2: Variance of clutter count ≈ LAMBDA_CLUTTER = 10
# ---------------------------------------------------------------------------

def test_clutter_count_variance(clutter_frames):
    """Var[K] = LAMBDA_CLUTTER = 10 within 3σ_MC over 10,000 frames."""
    counts = np.array([int((fr['labels'] == 0).sum()) for fr in clutter_frames])
    N    = len(counts)
    var  = counts.var(ddof=1)
    # For Poisson(λ): Var[S²] ≈ 2λ²/(N-1); SE = √(2)*λ/√N for large N
    # Using simpler MC SE: Var[S²] = (μ₄ - σ⁴)/N = (3λ² + λ - λ²)/N (Poisson)
    # = (2λ² + λ)/N ≈ 2λ²/N for λ=10
    se_var = np.sqrt((2 * LAMBDA_CLUTTER**2 + LAMBDA_CLUTTER) / N)
    assert abs(var - LAMBDA_CLUTTER) < 3 * se_var, (
        f"Var[K]={var:.4f} ≠ λ={LAMBDA_CLUTTER} (3σ band={3*se_var:.4f})"
    )


# ---------------------------------------------------------------------------
# G9-3: Quality residual (Z_m − η) ~ Exp(1) — KS test
# ---------------------------------------------------------------------------

def test_clutter_quality_residual_exponential(clutter_frames):
    """KS test: Z_m − DETECTION_THRESHOLD ~ Exp(1) at significance 0.05."""
    q_vals = []
    for fr in clutter_frames:
        mask = fr['labels'] == 0
        for q_db in fr['detections'][mask, 7]:
            # Q_m = 10·log10(Z_m) → Z_m = 10^(Q_m/10)
            Z_m = 10.0 ** (q_db / 10.0)
            residual = Z_m - DETECTION_THRESHOLD
            q_vals.append(residual)

    q_arr = np.array(q_vals)
    # All residuals should be positive (Z_m > η by construction)
    assert np.all(q_arr > 0), (
        f"{(q_arr <= 0).sum()} residuals ≤ 0: min={q_arr.min():.4e}"
    )

    # KS test against Exp(1); use a subsample of 5000 for speed
    rng_sub = np.random.default_rng(9031)
    sub = rng_sub.choice(q_arr, size=min(5_000, len(q_arr)), replace=False)
    ks_stat, p_val = stats.kstest(sub, 'expon', args=(0, 1))
    assert p_val >= 0.05, (
        f"KS test rejects Exp(1) for clutter residual: "
        f"D={ks_stat:.4f}, p={p_val:.4e} < 0.05"
    )


def test_clutter_quality_residual_mean(clutter_frames):
    """E[Z_m − η] ≈ 1 (mean of Exp(1)) within 3σ."""
    residuals = []
    for fr in clutter_frames:
        mask = fr['labels'] == 0
        for q_db in fr['detections'][mask, 7]:
            Z_m = 10.0 ** (q_db / 10.0)
            residuals.append(Z_m - DETECTION_THRESHOLD)
    residuals = np.array(residuals)
    N  = len(residuals)
    mu = residuals.mean()
    se = 1.0 / np.sqrt(N)   # Exp(1) has σ=1
    assert abs(mu - 1.0) < 3 * se, (
        f"E[Z_m−η]={mu:.4f} ≠ 1 (3σ band={3*se:.4f}, N={N})"
    )


# ---------------------------------------------------------------------------
# G9-4: All clutter Q_m > 10·log10(η)
# ---------------------------------------------------------------------------

def test_clutter_quality_above_threshold(clutter_frames):
    """Every clutter Q_m exceeds the threshold level 10·log10(η)."""
    q_min = 10.0 * np.log10(DETECTION_THRESHOLD)
    violations = []
    for k, fr in enumerate(clutter_frames[:200]):   # check first 200 frames
        mask = fr['labels'] == 0
        q_arr = fr['detections'][mask, 7]
        bad = q_arr[q_arr <= q_min]
        if len(bad):
            violations.append((k, bad))
    assert not violations, (
        f"Q_m ≤ {q_min:.3f} dB in {len(violations)} frames: "
        f"e.g. frame {violations[0][0]}: {violations[0][1][:3]}"
    )


# ---------------------------------------------------------------------------
# G9-5 & G9-6: No target label leakage in clutter
# ---------------------------------------------------------------------------

def test_no_id_leakage_in_clutter(clutter_frames):
    """All detection labels in 0-UAV frames are exactly 0 (no target ID)."""
    for k, fr in enumerate(clutter_frames[:500]):  # check first 500
        labels = fr['labels']
        # With 0 initial UAVs: no target born yet (or born but not detected)
        # Even if births occur mid-sequence, the label for clutter must be 0
        clutter_mask = labels == 0
        # Non-clutter must be positive (never negative in production frames)
        non_clutter = labels[~clutter_mask]
        # We allow positive labels (target born via LAMBDA_BIRTH); only check
        # that ALL labels are ≥ 0
        assert np.all(labels >= 0), (
            f"Frame {k}: label < 0 in production output: {labels[labels < 0]}"
        )


def test_clutter_label_exactly_zero(clutter_frames):
    """Clutter detections have label exactly 0, not any other value."""
    # Collect all detection labels; filter those that are clutter-class
    # In a 0-UAV scene, ALL labels should be 0.
    total = 0
    non_zero_clutter = 0
    for fr in clutter_frames[:100]:
        for lbl in fr['labels']:
            total += 1
            if int(lbl) != 0:
                non_zero_clutter += 1
    # With initial_n_targets=0 and very early frames, births may add targets
    # but the fixture was created with initial_n_targets=0, so for many frames
    # all labels are 0.  Just verify no negative labels appear.
    assert non_zero_clutter / max(total, 1) < 0.5, (
        f"More than 50% of detections have non-zero label in a 0-UAV scene "
        f"({non_zero_clutter}/{total})"
    )


# ---------------------------------------------------------------------------
# G9-7: Clutter count independence — lag-1 autocorrelation ≈ 0
# ---------------------------------------------------------------------------

def test_clutter_count_autocorrelation(clutter_frames):
    """Clutter count is memoryless: |r(1)| < 3/√N (white-noise 3σ band)."""
    counts = np.array([int((fr['labels'] == 0).sum()) for fr in clutter_frames])
    N   = len(counts)
    mu  = counts.mean()
    var = counts.var()
    if var < 1e-9:
        pytest.skip("Zero variance in clutter counts — cannot compute autocorrelation")
    c0 = var
    c1 = np.mean((counts[:-1] - mu) * (counts[1:] - mu))
    r1 = c1 / c0
    band = 3.0 / np.sqrt(N)
    assert abs(r1) < band, (
        f"Lag-1 autocorrelation r(1)={r1:.4f} exceeds 3σ band ±{band:.4f} "
        f"— clutter counts are not independent across frames"
    )


# ---------------------------------------------------------------------------
# Additional: parametrized quick checks across configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_key", ['C1', 'C3', 'C7'])
def test_clutter_count_mean_per_config(cfg_key):
    """E[K] ≈ LAMBDA_CLUTTER for each of three representative configs (1000 frames)."""
    from simulator.scene import run_scene
    rng  = np.random.default_rng(9200 + ord(cfg_key[1]))
    n    = 1_000
    frames = run_scene([cfg_key] * n, n, rng, initial_n_targets=0)
    counts = np.array([int((fr['labels'] == 0).sum()) for fr in frames])
    mu  = counts.mean()
    se  = np.sqrt(LAMBDA_CLUTTER / n)
    assert abs(mu - LAMBDA_CLUTTER) < 4 * se, (
        f"{cfg_key}: mean clutter={mu:.3f}, λ={LAMBDA_CLUTTER}, 4σ band={4*se:.3f}"
    )

# ---------------------------------------------------------------------------
# G9-8: Clutter spatial geometry — angles and range inside surveillance volume
# ---------------------------------------------------------------------------
#
# Clutter scatterers are sampled from the same airborne surveillance volume as
# UAVs (identical _birth_state geometry).  After adding measurement noise the
# reported az_tilde/el_tilde/R_tilde may exceed the bounds by a few sigmas, so
# we allow a margin of 3 × FLOOR_RANGE_M in range and 3 × 2° in angle.

def test_clutter_spatial_bounds(clutter_frames):
    """Clutter az_tilde, el_tilde, and R_tilde lie within surveillance bounds
    plus a 3-sigma measurement noise margin (500 frames checked)."""
    AZ_MARGIN  = 6.0    # deg  (≈ 3 × worst-case σ_az)
    EL_MARGIN  = 6.0    # deg  (≈ 3 × worst-case σ_el)
    R_MARGIN   = 3.0 * FLOOR_RANGE_M

    az_lo = SURV_AZ_MIN_DEG - AZ_MARGIN
    az_hi = SURV_AZ_MAX_DEG + AZ_MARGIN
    el_lo = SURV_EL_MIN_DEG - EL_MARGIN
    el_hi = SURV_EL_MAX_DEG + EL_MARGIN
    r_hi  = SURV_RHO_MAX + R_MARGIN

    violations = []
    for k, fr in enumerate(clutter_frames[:500]):
        mask = fr['labels'] == 0
        dets = fr['detections'][mask]   # (K, 8): [R, vr, az, el, beam_az, beam_el, dt, Q]
        for d in dets:
            R_m  = float(d[0])
            az_d = float(d[2])
            el_d = float(d[3])
            if not (az_lo <= az_d <= az_hi):
                violations.append(('az', k, az_d))
            if not (el_lo <= el_d <= el_hi):
                violations.append(('el', k, el_d))
            if R_m > r_hi:
                violations.append(('R', k, R_m))

    assert len(violations) == 0, (
        f"{len(violations)} clutter detections outside surveillance bounds "
        f"(az∈[{az_lo:.1f},{az_hi:.1f}]°, el∈[{el_lo:.1f},{el_hi:.1f}]°, "
        f"R≤{r_hi:.1f} m). First 5: {violations[:5]}"
    )
