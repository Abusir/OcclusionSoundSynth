from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Iterable


def _color_for_id(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def _bbox_payload(points: Iterable[tuple[float, float, float]]) -> dict[str, list[float]]:
    pts = list(points)
    if not pts:
        return {"location": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0]}
    mins = [min(point[axis] for point in pts) for axis in range(3)]
    maxs = [max(point[axis] for point in pts) for axis in range(3)]
    return {
        "location": [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)],
        "size": [maxs[axis] - mins[axis] for axis in range(3)],
    }


def write_semantic_stage_for_obj(obj_path: Path, *, semantic_asset_kind: str = "obj_colored") -> Path:
    """Create a Habitat stage config with semantic labels from OBJ usemtl."""

    obj_path = obj_path.resolve()
    scene_id = obj_path.stem
    stage_path = obj_path.with_name(f"{scene_id}.stage_config.json")
    scn_path = obj_path.with_suffix(".scn")
    ply_path = obj_path.with_name(f"{scene_id}_semantic.ply")
    semantic_obj_path = obj_path.with_name(f"{scene_id}_semantic.obj")

    vertices: list[tuple[float, float, float]] = []
    semantic_vertices: list[tuple[float, float, float, int, int, int]] = []
    semantic_faces: list[tuple[int, int, int]] = []
    current_material = "default"
    material_ids: dict[str, int] = {}
    material_points: dict[str, list[tuple[float, float, float]]] = {}

    def material_id(name: str) -> int:
        if name not in material_ids:
            material_ids[name] = len(material_ids)
        return material_ids[name]

    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("usemtl "):
            current_material = line.split(maxsplit=1)[1].strip()
        elif line.startswith("f "):
            raw = [part.split("/")[0] for part in line.split()[1:]]
            indices = [int(item) - 1 for item in raw]
            if len(indices) < 3:
                continue
            triangles = [indices] if len(indices) == 3 else [
                [indices[0], indices[offset], indices[offset + 1]]
                for offset in range(1, len(indices) - 1)
            ]
            red, green, blue = _color_for_id(material_id(current_material))
            for tri in triangles:
                face = []
                for vertex_index in tri:
                    vertex = vertices[vertex_index]
                    material_points.setdefault(current_material, []).append(vertex)
                    semantic_vertices.append((*vertex, red, green, blue))
                    face.append(len(semantic_vertices) - 1)
                semantic_faces.append((face[0], face[1], face[2]))

    ply_header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(semantic_vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(semantic_faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    ) + "\n"
    with ply_path.open("wb") as handle:
        handle.write(ply_header.encode("ascii"))
        for x, y, z, red, green, blue in semantic_vertices:
            handle.write(struct.pack("<fffBBB", x, y, z, red, green, blue))
        for a, b, c in semantic_faces:
            handle.write(struct.pack("<Biii", 3, a, b, c))

    semantic_obj_lines = ["o semantic_scene"]
    for x, y, z, red, green, blue in semantic_vertices:
        semantic_obj_lines.append(
            f"v {x:.9g} {y:.9g} {z:.9g} {red / 255.0:.9g} {green / 255.0:.9g} {blue / 255.0:.9g}"
        )
    for a, b, c in semantic_faces:
        semantic_obj_lines.append(f"f {a + 1} {b + 1} {c + 1}")
    semantic_obj_path.write_text("\n".join(semantic_obj_lines) + "\n", encoding="utf-8")

    objects = [
        {
            "class_": label,
            "id": idx,
            **_bbox_payload(material_points.get(label, [])),
        }
        for label, idx in material_ids.items()
    ]
    scn_path.write_text(json.dumps({"objects": objects}, indent=2), encoding="utf-8")
    semantic_asset = semantic_obj_path.name if semantic_asset_kind == "obj_colored" else ply_path.name
    stage_config = {
        "render_asset": obj_path.name,
        "collision_asset": obj_path.name,
        "semantic_asset": semantic_asset,
        "semantic_descriptor_filename": scn_path.name,
        "has_semantic_textures": False,
    }
    stage_path.write_text(json.dumps(stage_config, indent=2), encoding="utf-8")
    return stage_path
