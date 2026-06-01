from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


SCENE_TYPES = {
    1: "baffle_room",
    2: "l_shape_corridor",
    3: "t_shape_corridor",
    4: "empty_room",
    5: "open_field",
    6: "obstacle_forest",
}


@dataclass(frozen=True)
class Scene2D:
    scene_id: str
    scene_index: int
    scene_type_id: int
    scene_type: str
    variant_index: int
    is_outdoor: bool
    height_m: float
    boundary: Polygon
    obstacles: list[Polygon] = field(default_factory=list)
    walkable_area: Polygon = field(default_factory=Polygon)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.boundary.bounds


def _clean(poly: Polygon) -> Polygon:
    cleaned = poly.buffer(0)
    if cleaned.geom_type == "MultiPolygon":
        cleaned = max(cleaned.geoms, key=lambda geom: geom.area)
    return cleaned


def _walkable(boundary: Polygon, obstacles: list[Polygon]) -> Polygon:
    if not obstacles:
        return _clean(boundary)
    return _clean(boundary.difference(unary_union(obstacles)))


def _scene(
    scene_type_id: int,
    variant_index: int,
    is_outdoor: bool,
    boundary: Polygon,
    obstacles: list[Polygon],
    params: dict[str, Any],
    height_m: float,
) -> Scene2D:
    scene_type = SCENE_TYPES[scene_type_id]
    scene_index = (scene_type_id - 1) * 10 + variant_index
    scene_id = f"scene_{scene_type_id:02d}_{scene_type}_v{variant_index:02d}"
    return Scene2D(
        scene_id=scene_id,
        scene_index=scene_index,
        scene_type_id=scene_type_id,
        scene_type=scene_type,
        variant_index=variant_index,
        is_outdoor=is_outdoor,
        height_m=height_m,
        boundary=_clean(boundary),
        obstacles=[_clean(obs) for obs in obstacles],
        walkable_area=_walkable(boundary, obstacles),
        params=params,
    )


def generate_scene_1_baffle(variant_index: int, rng: random.Random) -> Scene2D:
    width = rng.uniform(4.5, 7.5)
    length = rng.uniform(4.5, 7.5)
    height = rng.uniform(2.6, 3.8)
    thickness = rng.uniform(0.16, 0.28)
    x_center = width / 2 + rng.uniform(-0.22 * width, 0.22 * width)
    baffle_length = length * rng.uniform(0.58, 0.86)
    from_top = rng.choice([False, True])
    if from_top:
        baffle = box(x_center - thickness / 2, length - baffle_length, x_center + thickness / 2, length)
    else:
        baffle = box(x_center - thickness / 2, 0, x_center + thickness / 2, baffle_length)
    boundary = box(0, 0, width, length)
    return _scene(
        1,
        variant_index,
        False,
        boundary,
        [baffle.intersection(boundary)],
        {
            "width_m": width,
            "length_m": length,
            "wall_thickness_m": thickness,
            "baffle_from_top": from_top,
        },
        height,
    )


def generate_scene_2_l_shape(variant_index: int, rng: random.Random) -> Scene2D:
    main_w = rng.uniform(1.6, 3.2)
    main_l = rng.uniform(5.5, 10.0)
    branch_w = rng.uniform(1.6, 3.2)
    branch_l = rng.uniform(4.0, 8.0)
    height = rng.uniform(2.5, 3.5)
    main = box(0, 0, main_w, main_l)
    branch = box(0, 0, main_w + branch_l, branch_w)
    boundary = _clean(main.union(branch))
    return _scene(
        2,
        variant_index,
        False,
        boundary,
        [],
        {
            "main_width_m": main_w,
            "main_length_m": main_l,
            "branch_width_m": branch_w,
            "branch_length_m": branch_l,
        },
        height,
    )


def generate_scene_3_t_shape(variant_index: int, rng: random.Random) -> Scene2D:
    top_w = rng.uniform(6.0, 12.0)
    top_h = rng.uniform(1.6, 3.0)
    stem_w = rng.uniform(1.6, 3.0)
    stem_h = rng.uniform(4.0, 8.0)
    height = rng.uniform(2.5, 3.5)
    stem_x = top_w / 2 - stem_w / 2 + rng.uniform(-min(1.2, top_w * 0.12), min(1.2, top_w * 0.12))
    stem_x = max(0.2, min(stem_x, top_w - stem_w - 0.2))
    top = box(0, stem_h, top_w, stem_h + top_h)
    stem = box(stem_x, 0, stem_x + stem_w, stem_h)
    boundary = _clean(top.union(stem))
    return _scene(
        3,
        variant_index,
        False,
        boundary,
        [],
        {
            "top_width_m": top_w,
            "top_height_m": top_h,
            "stem_width_m": stem_w,
            "stem_height_m": stem_h,
            "stem_x_m": stem_x,
        },
        height,
    )


def generate_scene_4_empty_room(variant_index: int, rng: random.Random) -> Scene2D:
    width = rng.uniform(5.0, 12.0)
    length = rng.uniform(5.0, 12.0)
    height = rng.uniform(2.6, 4.0)
    return _scene(
        4,
        variant_index,
        False,
        box(0, 0, width, length),
        [],
        {"width_m": width, "length_m": length},
        height,
    )


def generate_scene_5_open_field(variant_index: int, rng: random.Random) -> Scene2D:
    width = rng.uniform(8.0, 18.0)
    length = rng.uniform(8.0, 18.0)
    height = rng.uniform(5.0, 8.0)
    return _scene(
        5,
        variant_index,
        True,
        box(0, 0, width, length),
        [],
        {"width_m": width, "length_m": length},
        height,
    )


def generate_scene_6_forest(variant_index: int, rng: random.Random) -> Scene2D:
    width = rng.uniform(10.0, 20.0)
    length = rng.uniform(10.0, 20.0)
    height = rng.uniform(5.0, 8.0)
    boundary = box(0, 0, width, length)
    target_count = rng.randint(6, 16)
    obstacles: list[Polygon] = []
    for _ in range(target_count * 12):
        if len(obstacles) >= target_count:
            break
        radius = rng.uniform(0.22, 0.85)
        cx = rng.uniform(radius + 0.4, width - radius - 0.4)
        cy = rng.uniform(radius + 0.4, length - radius - 0.4)
        candidate = Point(cx, cy).buffer(radius, resolution=18)
        candidate = affinity.scale(candidate, xfact=rng.uniform(0.8, 1.25), yfact=rng.uniform(0.8, 1.25))
        padded = candidate.buffer(0.35)
        if all(not padded.intersects(obs) for obs in obstacles):
            obstacles.append(candidate)
    return _scene(
        6,
        variant_index,
        True,
        boundary,
        obstacles,
        {"width_m": width, "length_m": length, "obstacle_count": len(obstacles)},
        height,
    )


def generate_all_scenes(variants_per_type: int = 10, seed: int = 20260415) -> list[Scene2D]:
    rng = random.Random(seed)
    scenes: list[Scene2D] = []
    generators = [
        generate_scene_1_baffle,
        generate_scene_2_l_shape,
        generate_scene_3_t_shape,
        generate_scene_4_empty_room,
        generate_scene_5_open_field,
        generate_scene_6_forest,
    ]
    for type_id, generator in enumerate(generators, start=1):
        for variant_index in range(variants_per_type):
            scene = generator(variant_index, rng)
            if scene.scene_type_id != type_id:
                raise RuntimeError(f"Generator mismatch for scene type {type_id}")
            scenes.append(scene)
    return scenes
