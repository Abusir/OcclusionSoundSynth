from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .scene_generator import Scene2D


@dataclass(frozen=True)
class AcousticPlacement:
    receiver_xyz: tuple[float, float, float]
    source_xyz: tuple[float, float, float]
    source_type: str
    distance_m: float
    azimuth_rad: float
    elevation_rad: float
    is_los: bool
    obstruction_count: int
    obstruction_types: tuple[str, ...]


def _largest_polygon(poly: Polygon) -> Polygon:
    if poly.geom_type == "Polygon":
        return poly
    return max(poly.geoms, key=lambda geom: geom.area)


def sample_walkable_point(scene: Scene2D, rng: random.Random, margin_m: float = 0.25) -> tuple[float, float]:
    safe_area = scene.walkable_area.buffer(-margin_m)
    if safe_area.is_empty:
        safe_area = scene.walkable_area
    safe_area = _largest_polygon(safe_area)
    minx, miny, maxx, maxy = safe_area.bounds
    for _ in range(5000):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if safe_area.contains(Point(x, y)):
            return float(x), float(y)
    point = safe_area.representative_point()
    return float(point.x), float(point.y)


def line_obstructions(scene: Scene2D, receiver_xy: tuple[float, float], source_xy: tuple[float, float]) -> list[str]:
    ray = LineString([receiver_xy, source_xy])
    hits: list[str] = []
    if not scene.walkable_area.buffer(1e-7).covers(ray):
        hits.append("wall_or_non_walkable_space")
    for idx, obstacle in enumerate(scene.obstacles):
        if ray.crosses(obstacle) or ray.within(obstacle) or ray.intersects(obstacle):
            hits.append(f"obstacle_{idx:02d}")
    return hits


def relative_angles(receiver_xyz: tuple[float, float, float], source_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    direction = np.array(source_xyz, dtype=np.float64) - np.array(receiver_xyz, dtype=np.float64)
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-9:
        return 0.0, 0.0, 0.0
    azimuth = float(math.atan2(direction[1], direction[0]))
    elevation = float(math.asin(direction[2] / distance))
    return distance, azimuth, elevation


def sample_placement(
    scene: Scene2D,
    rng: random.Random,
    source_types: list[str] | None = None,
    min_distance_m: float = 1.0,
    prefer_obstructed: bool = False,
) -> AcousticPlacement:
    source_types = source_types or ["fire", "others"]
    receiver_xy = sample_walkable_point(scene, rng, margin_m=0.35)
    receiver_z = rng.uniform(1.15, 1.65)

    best_candidate: AcousticPlacement | None = None
    for _ in range(1500):
        source_xy = sample_walkable_point(scene, rng, margin_m=0.35)
        source_z = rng.uniform(0.35, min(2.2, scene.height_m - 0.35))
        receiver_xyz = (receiver_xy[0], receiver_xy[1], receiver_z)
        source_xyz = (source_xy[0], source_xy[1], source_z)
        distance, azimuth, elevation = relative_angles(receiver_xyz, source_xyz)
        if distance >= min_distance_m:
            hits = line_obstructions(scene, receiver_xy, source_xy)
            candidate = AcousticPlacement(
                receiver_xyz=receiver_xyz,
                source_xyz=source_xyz,
                source_type=rng.choice(source_types),
                distance_m=distance,
                azimuth_rad=azimuth,
                elevation_rad=elevation,
                is_los=len(hits) == 0,
                obstruction_count=len(hits),
                obstruction_types=tuple(hits),
            )
            if best_candidate is None:
                best_candidate = candidate
            if prefer_obstructed and not candidate.is_los:
                return candidate
            if not prefer_obstructed:
                return candidate

    if best_candidate is not None:
        return best_candidate

    source_xy = sample_walkable_point(scene, rng, margin_m=0.15)
    receiver_xyz = (receiver_xy[0], receiver_xy[1], receiver_z)
    source_xyz = (source_xy[0], source_xy[1], 1.0)
    distance, azimuth, elevation = relative_angles(receiver_xyz, source_xyz)
    hits = line_obstructions(scene, receiver_xy, source_xy)
    return AcousticPlacement(
        receiver_xyz=receiver_xyz,
        source_xyz=source_xyz,
        source_type=rng.choice(source_types),
        distance_m=distance,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        is_los=len(hits) == 0,
        obstruction_count=len(hits),
        obstruction_types=tuple(hits),
    )
