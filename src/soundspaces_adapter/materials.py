from __future__ import annotations

from pathlib import Path
import json

from soundspaces_adapter.material_database import MATERIAL_SPECS, SCENE_MATERIAL_ASSIGNMENTS

DEFAULT_MATERIAL_MAP = {
    "material_specs": MATERIAL_SPECS,
    "scene_material_assignments": SCENE_MATERIAL_ASSIGNMENTS,
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
