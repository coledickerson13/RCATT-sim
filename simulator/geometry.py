"""
simulator/geometry.py
======================
Shared geometric helpers for the RCATT simulator.

Centralised here so scene.py and measurement_model.py stay in sync when
surveillance-volume parameters change.
"""
from __future__ import annotations

import numpy as np

from config.sim_config import (
    GNB_POS,
    SURV_RHO_MAX,
    SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG,
    SURV_EL_MIN_DEG, SURV_EL_MAX_DEG,
    UAV_ALT_MIN, UAV_ALT_MAX,
)


def target_in_fov(state: np.ndarray) -> bool:
    """Return True if a target is inside the surveillance volume.

    Checks all four constraints:
      • slant range from gNB ∈ [1 m, SURV_RHO_MAX]
      • azimuth (0° = North / +y, atan2(Δx, Δy)) ∈ [SURV_AZ_MIN_DEG, SURV_AZ_MAX_DEG]
      • elevation ∈ [SURV_EL_MIN_DEG, SURV_EL_MAX_DEG]
      • absolute UAV altitude z ∈ [UAV_ALT_MIN, UAV_ALT_MAX]

    Parameters
    ----------
    state : (6,) or (3,) array — first three elements are [x, y, z] position (m)

    Returns
    -------
    bool
    """
    pos = np.asarray(state[:3], dtype=float)
    delta = pos - GNB_POS
    R = float(np.linalg.norm(delta))
    if R < 1.0 or R > SURV_RHO_MAX:
        return False
    az = np.degrees(np.arctan2(delta[0], delta[1]))   # 0° = North (+y)
    el = np.degrees(np.arcsin(np.clip(delta[2] / R, -1.0, 1.0)))
    return bool(
        SURV_AZ_MIN_DEG <= az <= SURV_AZ_MAX_DEG
        and SURV_EL_MIN_DEG <= el <= SURV_EL_MAX_DEG
        and UAV_ALT_MIN <= pos[2] <= UAV_ALT_MAX
    )
