from __future__ import annotations

import json
from pathlib import Path


FREQUENCIES = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


def _curve(values: list[float]) -> list[float]:
    if len(values) != len(FREQUENCIES):
        raise ValueError("material curves must have one value per frequency band")
    out: list[float] = []
    for freq, value in zip(FREQUENCIES, values):
        out.extend([freq, float(value)])
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
    }


def occ_material_database() -> dict[str, object]:
    """Return an RLR/SoundSpaces material database for generated OCC scenes.

    RLR maps semantic category labels to these materials. The labels are chosen
    to match the material names emitted by the OBJ exporter and the planned
    semantic categories for generated meshes.
    """

    return {
        "materials": [
            _material(
                "Default",
                ["default"],
                [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
                [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            ),
            _material(
                "Hard Floor",
                ["floor"],
                [0.08, 0.09, 0.10, 0.12, 0.15, 0.18],
                [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
                [0.002, 0.002, 0.001, 0.001, 0.0005, 0.0005],
            ),
            _material(
                "Painted Wall",
                ["wall"],
                [0.12, 0.10, 0.08, 0.07, 0.06, 0.06],
                [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
                [0.030, 0.020, 0.010, 0.004, 0.002, 0.001],
            ),
            _material(
                "Ceiling Tile",
                ["ceiling"],
                [0.25, 0.30, 0.35, 0.45, 0.55, 0.60],
                [0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
                [0.010, 0.008, 0.005, 0.003, 0.001, 0.001],
            ),
            _material(
                "Solid Obstacle",
                ["obstacle"],
                [0.18, 0.16, 0.14, 0.12, 0.10, 0.10],
                [0.18, 0.20, 0.22, 0.25, 0.28, 0.30],
                [0.020, 0.015, 0.008, 0.004, 0.002, 0.001],
            ),
            _material(
                "Outdoor Absorber",
                ["outdoor_ceiling"],
                [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            ),
        ]
    }


def write_occ_material_database(path: Path) -> dict[str, object]:
    payload = occ_material_database()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
