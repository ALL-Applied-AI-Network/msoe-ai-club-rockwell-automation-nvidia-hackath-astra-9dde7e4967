"""TCP interpolation and hard safety gates for policy 2."""

from __future__ import annotations

import math

import numpy as np


# Keep this aligned with the current official mock/cell interface while CSI
# reviews the requested deeper limit: -334 is allowed; -335 remains rejected.
TCP_BOX = np.array([[150.0, 650.0], [-450.0, 450.0], [-334.0, 300.0]])
# Exact policy-2 scene: objects occupy x=320..480, y=-190..200 and the four
# bins occupy x=205, y=-285..285. This remains well inside the official cell
# box and the separate 550 mm radius guard below.
DEMO_XY_BOX = np.array([[170.0, 520.0], [-310.0, 310.0]])
MAX_RADIUS_MM = 550.0


def safe_pose(pose) -> np.ndarray:
    value = np.asarray(pose, dtype=float).copy()
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid TCP target {value}")
    for axis in range(3):
        if not TCP_BOX[axis, 0] <= value[axis] <= TCP_BOX[axis, 1]:
            raise ValueError(f"TCP axis {axis} outside official cell limits: {value}")
    if not DEMO_XY_BOX[0, 0] <= value[0] <= DEMO_XY_BOX[0, 1]:
        raise ValueError(f"TCP x outside the demo work area: {value[0]:.1f} mm")
    if not DEMO_XY_BOX[1, 0] <= value[1] <= DEMO_XY_BOX[1, 1]:
        raise ValueError(f"TCP y outside the demo work area: {value[1]:.1f} mm")
    if math.hypot(value[0], value[1]) > MAX_RADIUS_MM:
        raise ValueError(f"TCP target beyond the 0.55 m reach guard: {value[:2]}")
    if np.max(np.abs(value[3:6])) > 360.0:
        raise ValueError("TCP orientation is outside +/-360 degrees")
    value[6] = float(value[6] >= 0.5)
    return value


def _angle_delta(start: float, finish: float) -> float:
    return (finish - start + 180.0) % 360.0 - 180.0


def interpolate(start, finish, max_linear_mm=5.0, max_angular_deg=3.0):
    a = safe_pose(start)
    b = safe_pose(finish)
    linear = float(np.linalg.norm(b[:3] - a[:3]))
    angular = np.array([_angle_delta(a[i], b[i]) for i in range(3, 6)])
    steps = max(
        1,
        int(math.ceil(linear / max_linear_mm)),
        int(math.ceil(float(np.max(np.abs(angular))) / max_angular_deg)),
    )
    if steps > 100:
        raise ValueError(f"Trajectory requires {steps} rows; split the motion")
    rows = []
    for index in range(1, steps + 1):
        fraction = index / steps
        row = a + (b - a) * fraction
        row[3:6] = a[3:6] + angular * fraction
        row[6] = a[6] if index < steps else b[6]
        rows.append(safe_pose(row).tolist())
    return rows


def dwell(pose, seconds: float, dt: float = 0.1):
    row = safe_pose(pose).tolist()
    return [list(row) for _ in range(max(1, int(round(seconds / dt))))]
