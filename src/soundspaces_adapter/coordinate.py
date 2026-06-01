from __future__ import annotations

from typing import Iterable

import numpy as np


def occ_to_habitat(point_xyz: Iterable[float]) -> np.ndarray:
    """Convert OCC coordinates to Habitat/SoundSpaces coordinates.

    OCC uses ``(x, y, z)`` where ``z`` is height. The OBJ exporter writes
    vertices as ``v x z y`` so Habitat reads a Y-up scene. The matching point
    transform is therefore ``(x, z, y)``.
    """

    x, y, z = [float(v) for v in point_xyz]
    return np.array([x, z, y], dtype=np.float32)


def habitat_to_occ(point_xyz: Iterable[float]) -> np.ndarray:
    """Convert Habitat/SoundSpaces coordinates back to OCC coordinates."""

    x, y, z = [float(v) for v in point_xyz]
    return np.array([x, z, y], dtype=np.float32)


def distance_m(a_xyz: Iterable[float], b_xyz: Iterable[float]) -> float:
    a = np.asarray(list(a_xyz), dtype=np.float64)
    b = np.asarray(list(b_xyz), dtype=np.float64)
    return float(np.linalg.norm(a - b))


def assert_round_trip(point_xyz: Iterable[float], atol: float = 1e-6) -> None:
    point = np.asarray(list(point_xyz), dtype=np.float32)
    recovered = habitat_to_occ(occ_to_habitat(point))
    if not np.allclose(point, recovered, atol=atol):
        raise AssertionError(f"coordinate round trip failed: {point.tolist()} -> {recovered.tolist()}")
