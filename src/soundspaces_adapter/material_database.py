from __future__ import annotations

import json
from pathlib import Path


FREQUENCIES = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
DAMPING_FREQUENCIES = [
    22.27948,
    27.64745,
    34.30876,
    42.57504,
    52.83297,
    65.56244,
    81.35888,
    100.96133,
    125.28674,
    155.47299,
    192.93234,
    239.41707,
    297.10159,
    368.68466,
    457.51477,
    567.74719,
    704.53906,
    874.28882,
    1084.93823,
    1346.34106,
    1670.72485,
    2073.26587,
    2572.79443,
    3192.67676,
    3961.91577,
    4916.48926,
    6101.05518,
    7571.03467,
    9395.17969,
    11658.83008,
    14467.89160,
    17953.74805,
]
DEFAULT_MEDIUM_DENSITY = 998.6546630859375
DEFAULT_MEDIUM_SPEED = 1483.9610595703125


MATERIAL_SPECS: dict[str, dict[str, object]] = {
    "default": {
        "name": "Default",
        "labels": ["default"],
        "description": "RLR/SoundSpaces default material.",
        "source_material": "Default",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        "scattering": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
        "transmission": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    "indoor_floor_hard": {
        "name": "Indoor Floor - Wood On Concrete",
        "labels": ["indoor_floor_hard", "floor"],
        "description": "RLR/SoundSpaces material: Wood On Concrete.",
        "source_material": "Wood On Concrete",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],
        "scattering": [0.10, 0.10, 0.10, 0.10, 0.10, 0.15],
        "transmission": [0.0040, 0.0079, 0.0056, 0.0016, 0.0014, 0.0005],
    },
    "outdoor_ground_grass": {
        "name": "Outdoor Ground - Grass",
        "labels": ["outdoor_ground_grass", "grass", "ground"],
        "description": "RLR/SoundSpaces material: Grass.",
        "source_material": "Grass",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.11, 0.26, 0.60, 0.69, 0.92, 0.99],
        "scattering": [0.30, 0.30, 0.40, 0.50, 0.60, 0.70],
        "transmission": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    "outdoor_ground_soil": {
        "name": "Outdoor Ground - Soil",
        "labels": ["outdoor_ground_soil", "soil"],
        "description": "RLR/SoundSpaces material: Soil.",
        "source_material": "Soil",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.15, 0.25, 0.40, 0.55, 0.60, 0.60],
        "scattering": [0.10, 0.20, 0.25, 0.40, 0.55, 0.70],
        "transmission": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
    "indoor_wall_reflective": {
        "name": "Indoor Wall - Gypsum Board",
        "labels": ["indoor_wall_reflective", "wall"],
        "description": "RLR/SoundSpaces material: Gypsum Board.",
        "source_material": "Gypsum Board",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
        "scattering": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
        "transmission": [0.0350, 0.0125, 0.0056, 0.0025, 0.0013, 0.0032],
    },
    "indoor_ceiling_reflective": {
        "name": "Indoor Ceiling - Acoustic Tile",
        "labels": ["indoor_ceiling_reflective", "ceiling"],
        "description": "RLR/SoundSpaces material: Acoustic Tile.",
        "source_material": "Acoustic Tile",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
        "scattering": [0.10, 0.15, 0.20, 0.20, 0.25, 0.30],
        "transmission": [0.050, 0.040, 0.030, 0.020, 0.005, 0.002],
    },
    "solid_occluder": {
        "name": "Solid Occluder - Thick Wood",
        "labels": ["solid_occluder", "obstacle"],
        "description": "RLR/SoundSpaces material: wood, Thick.",
        "source_material": "wood, Thick",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [0.19, 0.14, 0.09, 0.06, 0.06, 0.05],
        "scattering": [0.10, 0.10, 0.10, 0.10, 0.10, 0.15],
        "transmission": [0.035, 0.028, 0.028, 0.028, 0.0011, 0.0071],
    },
    "sky_absorber": {
        "name": "Open Boundary - Sound Proof",
        "labels": ["sky_absorber", "outdoor_ceiling", "open_boundary", "open_ceiling"],
        "description": "RLR/SoundSpaces material: Sound Proof, used as semantic absorber for open boundaries.",
        "source_material": "Sound Proof",
        "source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
        "absorption": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        "scattering": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        "transmission": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    },
}


SCENE_MATERIAL_ASSIGNMENTS: dict[str, dict[str, str | None]] = {
    "baffle_room": {
        "floor": "indoor_floor_hard",
        "wall": "indoor_wall_reflective",
        "ceiling": "indoor_wall_reflective",
        "obstacle": "solid_occluder",
        "open_boundary": None,
        "open_ceiling": None,
    },
    "l_shape_corridor": {
        "floor": "indoor_floor_hard",
        "wall": "indoor_wall_reflective",
        "ceiling": "indoor_wall_reflective",
        "obstacle": None,
        "open_boundary": None,
        "open_ceiling": None,
    },
    "t_shape_corridor": {
        "floor": "indoor_floor_hard",
        "wall": "indoor_wall_reflective",
        "ceiling": "indoor_wall_reflective",
        "obstacle": None,
        "open_boundary": None,
        "open_ceiling": None,
    },
    "empty_room": {
        "floor": "indoor_floor_hard",
        "wall": "indoor_wall_reflective",
        "ceiling": "indoor_wall_reflective",
        "obstacle": None,
        "open_boundary": None,
        "open_ceiling": None,
    },
    "open_field": {
        "floor": "outdoor_ground_grass",
        "wall": None,
        "ceiling": None,
        "obstacle": None,
        "open_boundary": "sky_absorber",
        "open_ceiling": "sky_absorber",
    },
    "obstacle_forest": {
        "floor": "outdoor_ground_soil",
        "wall": None,
        "ceiling": None,
        "obstacle": "solid_occluder",
        "open_boundary": "sky_absorber",
        "open_ceiling": "sky_absorber",
    },
}


def _curve(values: list[float]) -> list[float]:
    if len(values) != len(FREQUENCIES):
        raise ValueError("material curves must have one value per frequency band")
    out: list[float] = []
    for freq, value in zip(FREQUENCIES, values):
        out.extend([freq, float(value)])
    return out


def _damping_curve() -> list[float]:
    out: list[float] = []
    base = 1.1595274e-10
    for idx, freq in enumerate(DAMPING_FREQUENCIES):
        out.extend([freq, base * (1.54**idx)])
    return out


def _material(
    name: str,
    labels: list[str],
    absorption: list[float],
    scattering: list[float],
    transmission: list[float],
) -> dict[str, object]:
    return {
        "name": name,
        "labels": labels,
        "absorption": _curve(absorption),
        "scattering": _curve(scattering),
        "transmission": _curve(transmission),
        "damping": _damping_curve(),
        "density": DEFAULT_MEDIUM_DENSITY,
        "speed": DEFAULT_MEDIUM_SPEED,
    }


def occ_material_database() -> dict[str, object]:
    """Return an RLR/SoundSpaces material database for generated OCC scenes.

    RLR maps semantic category labels to these materials. The labels are chosen
    to match the material names emitted by the OBJ exporter and the planned
    semantic categories for generated meshes.
    """

    return {
        "materials": [_material_from_spec(spec) for spec in MATERIAL_SPECS.values()],
        "scene_material_assignments": SCENE_MATERIAL_ASSIGNMENTS,
    }


def _material_from_spec(spec: dict[str, object]) -> dict[str, object]:
    return _material(
        str(spec["name"]),
        list(spec["labels"]),  # type: ignore[arg-type]
        list(spec["absorption"]),  # type: ignore[arg-type]
        list(spec["scattering"]),  # type: ignore[arg-type]
        list(spec["transmission"]),  # type: ignore[arg-type]
    )


def scene_material_assignment(scene_type: str) -> dict[str, str | None]:
    if scene_type not in SCENE_MATERIAL_ASSIGNMENTS:
        raise KeyError(f"unknown scene type: {scene_type}")
    return dict(SCENE_MATERIAL_ASSIGNMENTS[scene_type])


def scene_material_report(scene_type: str) -> dict[str, object]:
    assignment = scene_material_assignment(scene_type)
    materials = {
        key: MATERIAL_SPECS[value]
        for key, value in assignment.items()
        if value is not None
    }
    return {"scene_type": scene_type, "assignment": assignment, "materials": materials}


def write_occ_material_database(path: Path) -> dict[str, object]:
    payload = occ_material_database()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_scene_material_assignments(path: Path) -> dict[str, object]:
    payload = {
        "material_specs": MATERIAL_SPECS,
        "scene_material_assignments": SCENE_MATERIAL_ASSIGNMENTS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
