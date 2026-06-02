from __future__ import annotations

import json
from pathlib import Path
import struct


def _color_for_id(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 255, (value >> 8) & 255, value & 255)


def write_semantic_stage_for_obj(obj_path: Path) -> Path:
    """Create a Habitat stage config with semantic labels from OBJ usemtl."""

    obj_path = obj_path.resolve()
    scene_id = obj_path.stem
    stage_path = obj_path.with_name(f"{scene_id}.stage_config.json")
    scn_path = obj_path.with_suffix(".scn")
    ply_path = obj_path.with_name(f"{scene_id}_semantic.ply")

    vertices: list[tuple[float, float, float]] = []
    semantic_vertices: list[tuple[float, float, float, int, int, int]] = []
    semantic_faces: list[tuple[int, int, int]] = []
    current_material = "default"
    material_ids: dict[str, int] = {}

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
                    semantic_vertices.append((*vertices[vertex_index], red, green, blue))
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

    objects = [{"class_": label, "id": idx} for label, idx in material_ids.items()]
    scn_path.write_text(json.dumps({"objects": objects}, indent=2), encoding="utf-8")
    stage_config = {
        "render_asset": obj_path.name,
        "collision_asset": obj_path.name,
        "semantic_asset": ply_path.name,
        "semantic_descriptor_filename": scn_path.name,
        "has_semantic_textures": False,
    }
    stage_path.write_text(json.dumps(stage_config, indent=2), encoding="utf-8")
    return stage_path
