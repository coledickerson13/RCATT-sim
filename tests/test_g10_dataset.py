"""
test_g10_dataset.py – G10: HDF5 dataset integrity
==================================================
Generates a small pilot dataset and validates the HDF5 file structure,
label semantics, stochastic process statistics, and reproducibility.

Pilot parameters (small enough for CI: ~90-120 s total):
  N_TRAIN  = 40 sequences
  N_VAL    = 20 sequences  (in-distribution, seed_offset=1)
  N_TEST   = 20 sequences  (in-distribution, seed_offset=2)
  N_OOD    = 20 sequences  (OOD configs,     seed_offset=3)
  T        = N_FRAMES_PER_SEQ = 20

API (from simulator.generate_dataset):
  generate_split(split_name, n_sequences, cfg_pool, out_path, base_seed, n_workers)

HDF5 layout (confirmed from generate_dataset.py)
-------------------------------------------------
  contexts    (N, T, 5)          float32
  detections  (N, T, M_MAX, 8)   float32
  det_valid   (N, T, M_MAX)      bool
  labels      (N, T, M_MAX)      int32   (-1=pad, 0=clutter, >0=UAV ID)
  states      (N, T, N_MAX, 6)   float32
  state_valid (N, T, N_MAX)      bool
  visible     (N, T, N_MAX)      bool
  state_ids   (N, T, N_MAX)      int32   (-1=padded slot)
  cfg_keys    (N, T)             uint8   index into CONFIG_KEY_LIST
  sic_dbs     (N, T)             float32

Attributes: split, n_sequences, T, M_max, N_max, config_key_list,
            cfg_pool, base_seed, generated_at.

Pass criteria
-------------
G10-1  HDF5 shape: all 10 datasets have exactly the right shape.
G10-2  det_valid == (labels != -1) for every element.
G10-3  Labels: exactly {-1, 0} ∪ positive integers; every positive label
       has a matching state_ids entry with state_valid True.
G10-4  state_valid True entries count ≤ N_MAX; state_ids > 0 where valid.
G10-5  Resource config self-transition ≈ 0.85 (Markov chain in cfg_keys).
G10-6  OOD split's cfg_pool contains only OOD configs; train uses no OOD.
G10-7  Reproducibility: same seed → identical detections / labels arrays.
"""

import sys
import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from config.sim_config import RESOURCE_CONFIGS

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")

# ---------------------------------------------------------------------------
# Constants matching generate_dataset.py
# ---------------------------------------------------------------------------

from simulator.generate_dataset import (
    TRAIN_CONFIGS, OOD_CONFIGS, CONFIG_KEY_LIST, M_MAX,
)
from config.sim_config import N_FRAMES_PER_SEQ, N_TARGET_MAX

_N_MAX = N_TARGET_MAX          # 5
_T     = N_FRAMES_PER_SEQ     # 20

_PILOT = {
    'train': {'n': 40, 'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 0},
    'val':   {'n': 20, 'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 1},
    'test':  {'n': 20, 'cfg_pool': TRAIN_CONFIGS, 'seed_offset': 2},
    'ood':   {'n': 20, 'cfg_pool': OOD_CONFIGS,   'seed_offset': 3},
}
_BASE_SEED = 999   # arbitrary; different from production seed=0


# ---------------------------------------------------------------------------
# Session-scoped fixture: generate pilot splits once
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def pilot_dir(tmp_path_factory):
    """Generate pilot HDF5 files for train, val, test, and ood splits."""
    from simulator.generate_dataset import generate_split
    out = tmp_path_factory.mktemp('pilot')
    for split, cfg in _PILOT.items():
        path = str(out / f'{split}.h5')
        generate_split(
            split_name  = split,
            n_sequences = cfg['n'],
            cfg_pool    = cfg['cfg_pool'],
            out_path    = path,
            base_seed   = _BASE_SEED * 10 + cfg['seed_offset'],
            n_workers   = 1,
        )
    return out


# ---------------------------------------------------------------------------
# G10-1: HDF5 shape checks
# ---------------------------------------------------------------------------

class TestShapes:
    @pytest.mark.parametrize("split", list(_PILOT))
    def test_shapes(self, pilot_dir, split):
        """All datasets have the expected shapes."""
        N   = _PILOT[split]['n']
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            shapes = {k: f[k].shape for k in f if isinstance(f[k], h5py.Dataset)}

        expected = {
            'detections':  (N, _T, M_MAX, 8),
            'contexts':    (N, _T, 5),
            'det_valid':   (N, _T, M_MAX),
            'labels':      (N, _T, M_MAX),
            'states':      (N, _T, _N_MAX, 6),
            'state_valid': (N, _T, _N_MAX),
            'visible':     (N, _T, _N_MAX),
            'state_ids':   (N, _T, _N_MAX),
            'cfg_keys':    (N, _T),
            'sic_dbs':     (N, _T),
        }
        for key, exp in expected.items():
            assert key in shapes, f"split={split}: missing '{key}'"
            assert shapes[key] == exp, (
                f"split={split}: '{key}' shape {shapes[key]} ≠ {exp}"
            )


# ---------------------------------------------------------------------------
# G10-2: det_valid == (labels != -1)
# ---------------------------------------------------------------------------

class TestDetValid:
    @pytest.mark.parametrize("split", list(_PILOT))
    def test_det_valid_matches_labels(self, pilot_dir, split):
        """det_valid[s,t,m] == (labels[s,t,m] != -1) for all entries."""
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            labels    = f['labels'][:]
            det_valid = f['det_valid'][:]
        expected = labels != -1
        mismatch = int(np.sum(det_valid != expected))
        assert mismatch == 0, f"split={split}: {mismatch} det_valid/label mismatches"


# ---------------------------------------------------------------------------
# G10-3: Label validity and state consistency
# ---------------------------------------------------------------------------

class TestLabels:
    @pytest.mark.parametrize("split", list(_PILOT))
    def test_label_set(self, pilot_dir, split):
        """All labels are -1, 0, or positive integers."""
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            labels = f['labels'][:]
        bad = labels[(labels < -1)]   # labels < -1 are invalid
        assert len(bad) == 0, (
            f"split={split}: {len(bad)} labels < -1: {np.unique(bad)[:5]}"
        )
        # -1 is padding; 0 is clutter; positive is UAV ID
        in_neg1_0_pos = np.all((labels == -1) | (labels == 0) | (labels > 0))
        assert in_neg1_0_pos  # always true if no labels < -1

    @pytest.mark.parametrize("split", list(_PILOT))
    def test_positive_label_has_valid_state(self, pilot_dir, split):
        """Every positive label has a matching state_ids entry with state_valid."""
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            labels      = f['labels'][:]        # (N,T,M)
            state_valid = f['state_valid'][:]   # (N,T,N_MAX) bool
            state_ids   = f['state_ids'][:]     # (N,T,N_MAX) int32

        violations = 0
        for s in range(labels.shape[0]):
            for t in range(labels.shape[1]):
                pos = labels[s, t][labels[s, t] > 0]
                for lbl in pos:
                    found = np.any((state_ids[s, t] == lbl) & state_valid[s, t])
                    if not found:
                        violations += 1

        assert violations == 0, (
            f"split={split}: {violations} positive labels with no valid state entry"
        )


# ---------------------------------------------------------------------------
# G10-4: state_valid and state_ids consistency
# ---------------------------------------------------------------------------

class TestStateValid:
    @pytest.mark.parametrize("split", list(_PILOT))
    def test_state_count_bounded(self, pilot_dir, split):
        """state_valid count per frame slot ≤ N_MAX."""
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            sv = f['state_valid'][:]
        counts = sv.sum(axis=-1)
        assert int(counts.max()) <= _N_MAX, (
            f"split={split}: max state_valid count {counts.max()} > N_MAX={_N_MAX}"
        )

    @pytest.mark.parametrize("split", list(_PILOT))
    def test_state_ids_positive_where_valid(self, pilot_dir, split):
        """state_ids > 0 wherever state_valid is True (slot is occupied)."""
        path = str(pilot_dir / f'{split}.h5')
        with h5py.File(path, 'r') as f:
            sv  = f['state_valid'][:]
            ids = f['state_ids'][:]
        bad = ids[sv] <= 0
        assert not np.any(bad), (
            f"split={split}: {bad.sum()} state_ids ≤ 0 where state_valid=True"
        )


# ---------------------------------------------------------------------------
# G10-5: Resource config self-transition ≈ 0.85
# ---------------------------------------------------------------------------

class TestTransitions:
    def test_resource_self_transition_train(self, pilot_dir):
        """Resource cfg_keys self-transition rate ≈ 0.85 (train split)."""
        path = str(pilot_dir / 'train.h5')
        with h5py.File(path, 'r') as f:
            ck = f['cfg_keys'][:]   # (N, T) uint8 indices

        n_self, n_total = 0, 0
        for s in range(ck.shape[0]):
            for t in range(1, ck.shape[1]):
                if ck[s, t] == ck[s, t - 1]:
                    n_self += 1
                n_total += 1

        assert n_total >= 20, f"Too few transitions ({n_total}) to test"
        rate = n_self / n_total
        # Allow ±10% due to small pilot size
        assert 0.75 <= rate <= 0.95, (
            f"Resource self-transition={rate:.3f} outside [0.75, 0.95] "
            f"({n_self}/{n_total})"
        )


# ---------------------------------------------------------------------------
# G10-6: OOD contamination check
# ---------------------------------------------------------------------------

class TestOODContamination:
    def test_ood_uses_only_ood_configs(self, pilot_dir):
        """OOD split's cfg_pool attribute contains only OOD config keys."""
        path = str(pilot_dir / 'ood.h5')
        with h5py.File(path, 'r') as f:
            cfg_pool_raw = f.attrs.get('cfg_pool', None)
        assert cfg_pool_raw is not None, "OOD HDF5 missing 'cfg_pool' attribute"

        # cfg_pool may be stored as bytes, str, or list
        if hasattr(cfg_pool_raw, 'tolist'):
            cfg_pool = [str(x) for x in cfg_pool_raw.tolist()]
        else:
            import json
            cfg_pool = json.loads(cfg_pool_raw) if isinstance(cfg_pool_raw, str) else list(cfg_pool_raw)

        for key in cfg_pool:
            key = str(key)
            assert key in RESOURCE_CONFIGS, f"OOD cfg_pool has unknown key: {key}"
            assert RESOURCE_CONFIGS[key].get('split') == 'ood', (
                f"OOD split uses train config: {key}"
            )

    def test_train_uses_no_ood_configs(self, pilot_dir):
        """Train split's cfg_pool contains no OOD config keys."""
        path = str(pilot_dir / 'train.h5')
        with h5py.File(path, 'r') as f:
            cfg_pool_raw = f.attrs.get('cfg_pool', None)
        assert cfg_pool_raw is not None, "train HDF5 missing 'cfg_pool' attribute"

        if hasattr(cfg_pool_raw, 'tolist'):
            cfg_pool = [str(x) for x in cfg_pool_raw.tolist()]
        else:
            import json
            cfg_pool = json.loads(cfg_pool_raw) if isinstance(cfg_pool_raw, str) else list(cfg_pool_raw)

        for key in cfg_pool:
            key = str(key)
            if key in RESOURCE_CONFIGS:
                assert RESOURCE_CONFIGS[key].get('split') != 'ood', (
                    f"Train split uses OOD config: {key}"
                )

    def test_cfg_key_indices_valid_in_train(self, pilot_dir):
        """cfg_keys uint8 indices in train HDF5 are valid CONFIG_KEY_LIST indices."""
        path = str(pilot_dir / 'train.h5')
        with h5py.File(path, 'r') as f:
            ck = f['cfg_keys'][:]
        max_idx = len(CONFIG_KEY_LIST) - 1
        assert int(ck.max()) <= max_idx, (
            f"cfg_keys max index {ck.max()} > {max_idx} (len(CONFIG_KEY_LIST)-1)"
        )


# ---------------------------------------------------------------------------
# G10-7: Reproducibility — same seed → identical data
# ---------------------------------------------------------------------------

class TestReproducibility:
    @pytest.mark.parametrize("split", list(_PILOT))
    def test_reproducible(self, pilot_dir, tmp_path, split):
        """Running generate_split twice with identical seed produces identical data."""
        from simulator.generate_dataset import generate_split

        cfg = _PILOT[split]
        path1 = str(tmp_path / f'{split}_run1.h5')
        path2 = str(tmp_path / f'{split}_run2.h5')
        seed = _BASE_SEED * 10 + cfg['seed_offset']
        N    = cfg['n']

        generate_split(split, N, cfg['cfg_pool'], path1, seed, n_workers=1)
        generate_split(split, N, cfg['cfg_pool'], path2, seed, n_workers=1)

        with h5py.File(path1, 'r') as f1, h5py.File(path2, 'r') as f2:
            for key in ('detections', 'labels', 'states', 'det_valid',
                        'state_valid', 'state_ids', 'cfg_keys'):
                if key not in f1 or key not in f2:
                    continue
                d1, d2 = f1[key][:], f2[key][:]
                assert np.array_equal(d1, d2), (
                    f"split={split}: '{key}' differs between two identical runs"
                )
