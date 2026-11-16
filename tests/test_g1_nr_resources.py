"""G1: NR resource configuration and PRS timing validation.

Independent verification of all eight configurations against the frozen
simulation specification.  No production functions are called for the
reference values; they are computed from first principles here.

Pass criteria
-------------
- Pilot RE counts match the reference table exactly (integer comparison).
- Comb-2 structure: 4 PRS symbols per slot at positions k ∈ {0,1,2,3}.
- CPI timing: n_slot × SLOT_DURATION, correct for all 8 configs.
- t_obs independently reconstructed to within numerical precision.
- T_SCAN = N_BEAMS × CPI ≤ T_FRAME = 500 ms for all configs.
- C5 and C7 classified as OOD; C1,C2,C3,C4,C6,C8 as train.
"""

import pytest
import numpy as np

from config.sim_config import (
    RESOURCE_CONFIGS, N_BEAMS, SLOT_DURATION, N_SYM_PER_SLOT, COMB_K,
    N_PIL_PER_RB_PER_SLOT, T_FRAME,
)

# ---------------------------------------------------------------------------
# Reference table (from frozen measurement-generation reference)
# key → (n_sc_carrier, n_rb_prs, n_slot, cpi_ms, n_pil)
# ---------------------------------------------------------------------------
REFERENCE = {
    'C1': (612,    48,  8,  4.0,  9_216),
    'C2': (1272,  104,  8,  4.0, 19_968),
    'C3': (612,    48, 16,  8.0, 18_432),
    'C4': (1272,  104, 16,  8.0, 39_936),
    'C5': (288,    24,  8,  4.0,  4_608),
    'C6': (3276,  272,  8,  4.0, 52_224),
    'C7': (612,    48,  4,  2.0,  4_608),
    'C8': (1596,  132, 16,  8.0, 50_688),
}

# ---------------------------------------------------------------------------
# Independent reference calculations (first principles)
# ---------------------------------------------------------------------------

def _ref_n_pil(n_rb_prs: int, n_slot: int) -> int:
    """Pilot RE count from Comb-2 first principles.

    Comb-2 PRS: 12 sc/RB × (1/Comb_K) = 6 occupied sc/RB per symbol,
    4 PRS symbols per slot (k=0,1,2,3).
    Total = 6 × 4 × n_rb_prs × n_slot
    """
    prs_syms_per_slot = 4                           # k = 0,1,2,3
    occupied_sc_per_rb_per_sym = 12 // COMB_K       # = 6
    return occupied_sc_per_rb_per_sym * prs_syms_per_slot * n_rb_prs * n_slot


def _ref_t_obs(n_slot: int) -> float:
    """First-to-last pilot time span (s), independent of production code.

    PRS symbols are at k=0,1,2,3 within each of n_slot slots.
    Last pilot: slot n_slot-1, symbol 3
    t = ((n_slot-1)*14 + 3) × T_sym    where T_sym = SLOT_DURATION/N_SYM_PER_SLOT
    """
    T_sym = SLOT_DURATION / N_SYM_PER_SLOT
    return ((n_slot - 1) * N_SYM_PER_SLOT + 3) * T_sym


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_carrier_subcarriers(key, ref):
    """Carrier subcarrier count matches reference table."""
    n_sc_ref = ref[0]
    assert RESOURCE_CONFIGS[key]['n_sc_carrier'] == n_sc_ref, (
        f"{key}: n_sc_carrier={RESOURCE_CONFIGS[key]['n_sc_carrier']}, expected {n_sc_ref}"
    )


@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_prs_rb_count(key, ref):
    """PRS RB count matches reference table."""
    n_rb_ref = ref[1]
    assert RESOURCE_CONFIGS[key]['n_rb_prs'] == n_rb_ref


@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_n_slot(key, ref):
    """n_slot matches reference table."""
    n_slot_ref = ref[2]
    assert RESOURCE_CONFIGS[key]['n_slot'] == n_slot_ref


@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_cpi_ms(key, ref):
    """CPI (cpi_s) matches reference in milliseconds."""
    cpi_ms_ref = ref[3]
    cpi_ms = RESOURCE_CONFIGS[key]['cpi_s'] * 1e3
    assert abs(cpi_ms - cpi_ms_ref) < 1e-9, (
        f"{key}: CPI={cpi_ms:.6f} ms, expected {cpi_ms_ref:.6f} ms"
    )


@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_pilot_re_count_matches_reference(key, ref):
    """n_pil from RESOURCE_CONFIGS matches the frozen reference table exactly."""
    n_pil_ref = ref[4]
    assert RESOURCE_CONFIGS[key]['n_pil'] == n_pil_ref, (
        f"{key}: n_pil={RESOURCE_CONFIGS[key]['n_pil']}, expected {n_pil_ref}"
    )


@pytest.mark.parametrize("key,ref", REFERENCE.items())
def test_pilot_re_count_independent(key, ref):
    """n_pil matches independently computed value from Comb-2 first principles."""
    n_rb  = RESOURCE_CONFIGS[key]['n_rb_prs']
    n_slot = RESOURCE_CONFIGS[key]['n_slot']
    n_pil_ref = ref[4]
    n_pil_indep = _ref_n_pil(n_rb, n_slot)
    assert n_pil_indep == n_pil_ref, (
        f"{key}: independent={n_pil_indep}, reference={n_pil_ref}"
    )
    assert RESOURCE_CONFIGS[key]['n_pil'] == n_pil_indep


@pytest.mark.parametrize("key", RESOURCE_CONFIGS.keys())
def test_t_obs_independent(key):
    """t_obs matches independently computed first-to-last pilot time span."""
    n_slot = RESOURCE_CONFIGS[key]['n_slot']
    t_obs_ref = _ref_t_obs(n_slot)
    t_obs_sim = RESOURCE_CONFIGS[key]['t_obs_s']
    assert abs(t_obs_sim - t_obs_ref) < 1e-12, (
        f"{key}: t_obs={t_obs_sim*1e3:.6f} ms, expected {t_obs_ref*1e3:.6f} ms"
    )


@pytest.mark.parametrize("key", RESOURCE_CONFIGS.keys())
def test_scan_fits_in_frame(key):
    """T_SCAN = N_BEAMS × CPI ≤ T_FRAME = 500 ms."""
    cpi = RESOURCE_CONFIGS[key]['cpi_s']
    t_scan = N_BEAMS * cpi
    assert t_scan <= T_FRAME + 1e-9, (
        f"{key}: T_SCAN={t_scan*1e3:.1f} ms > T_FRAME={T_FRAME*1e3:.1f} ms"
    )


@pytest.mark.parametrize("key,expected_scan_ms", [
    ('C1', 128.0), ('C2', 128.0), ('C3', 256.0), ('C4', 256.0),
    ('C5', 128.0), ('C6', 128.0), ('C7',  64.0), ('C8', 256.0),
])
def test_scan_duration_exact(key, expected_scan_ms):
    """T_SCAN values match the expected 64/128/256 ms."""
    t_scan_ms = N_BEAMS * RESOURCE_CONFIGS[key]['cpi_s'] * 1e3
    assert abs(t_scan_ms - expected_scan_ms) < 1e-9, (
        f"{key}: T_SCAN={t_scan_ms:.3f} ms, expected {expected_scan_ms:.3f} ms"
    )


def test_ood_configs():
    """C5 and C7 are OOD; all others are train."""
    expected_ood   = {'C5', 'C7'}
    expected_train = {'C1', 'C2', 'C3', 'C4', 'C6', 'C8'}
    for key, cfg in RESOURCE_CONFIGS.items():
        expected = 'ood' if key in expected_ood else 'train'
        assert cfg['split'] == expected, f"{key}: split={cfg['split']}, expected {expected}"


def test_comb2_prs_symbols_per_slot():
    """The Comb-2 PRS symbol positions k=0,1,2,3 give 4 symbols per slot."""
    prs_sym_positions = [0, 1, 2, 3]  # defined in 3GPP TS 38.211
    n_sc_per_rb = 12
    occupied = n_sc_per_rb // COMB_K   # 6 per symbol per RB
    assert occupied * len(prs_sym_positions) == N_PIL_PER_RB_PER_SLOT, (
        f"24 pilots/RB/slot: got {occupied * len(prs_sym_positions)}"
    )


def test_all_configs_present():
    """All eight configurations C1–C8 are present."""
    expected = {'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'}
    assert set(RESOURCE_CONFIGS.keys()) == expected
