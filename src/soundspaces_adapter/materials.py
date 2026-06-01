from __future__ import annotations

from pathlib import Path
import json


DEFAULT_MATERIAL_MAP = {
    "floor": {
        "description": "hard floor, moderate reflection",
        "absorption": [0.18, 0.15, 0.12, 0.10],
        "scattering": 0.10,
    },
    "ceiling": {
        "description": "painted ceiling",
        "absorption": [0.25, 0.20, 0.18, 0.15],
        "scattering": 0.12,
    },
    "outdoor_ceiling": {
        "description": "semantic sky absorber for open scenes",
        "absorption": [0.95, 0.95, 0.95, 0.95],
        "scattering": 0.00,
    },
    "wall": {
        "description": "rigid wall or corridor boundary",
        "absorption": [0.12, 0.10, 0.08, 0.08],
        "scattering": 0.08,
    },
    "obstacle": {
        "description": "solid obstacle, baffle, tree trunk, or column",
        "absorption": [0.20, 0.17, 0.14, 0.12],
        "scattering": 0.18,
    },
}


def write_material_map(path: Path, material_map: dict[str, object] | None = None) -> dict[str, object]:
    """Write a human-readable material map next to exported OBJ/MTL files.

    SoundSpaces material APIs have changed across builds, so this file is a
    stable project-side record. The backend tries to enable materials when the
    installed Habitat-Sim build exposes the relevant API.
    """

    payload = material_map or DEFAULT_MATERIAL_MAP
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
