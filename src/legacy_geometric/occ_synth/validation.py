from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
from shapely.geometry import Point

from .scene_generator import Scene2D


def validate_scene(scene: Scene2D) -> list[str]:
    errors: list[str] = []
    if not scene.boundary.is_valid:
        errors.append(f"{scene.scene_id}: boundary is invalid")
    if scene.walkable_area.is_empty or scene.walkable_area.area < 1.0:
        errors.append(f"{scene.scene_id}: walkable area too small")
    for idx, obstacle in enumerate(scene.obstacles):
        if not obstacle.is_valid:
            errors.append(f"{scene.scene_id}: obstacle {idx} is invalid")
        if not scene.boundary.contains(obstacle.buffer(-1e-6)):
            errors.append(f"{scene.scene_id}: obstacle {idx} outside boundary")
    return errors


def validate_catalog(scenes: list[Scene2D], variants_per_type: int) -> list[str]:
    errors: list[str] = []
    expected = 6 * variants_per_type
    if len(scenes) != expected:
        errors.append(f"scene count mismatch: got {len(scenes)}, expected {expected}")
    counts = Counter(scene.scene_type_id for scene in scenes)
    for type_id in range(1, 7):
        if counts[type_id] != variants_per_type:
            errors.append(f"scene type {type_id} count mismatch: got {counts[type_id]}")
    for scene in scenes:
        errors.extend(validate_scene(scene))
    return errors


def validate_label(label: dict[str, Any], scene: Scene2D, allowed_source_classes: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    for key in ["scene_id", "scene_type", "source", "receiver", "relative", "occlusion", "files"]:
        if key not in label:
            errors.append(f"{scene.scene_id}: missing label key {key}")
    source_type = label.get("source", {}).get("type")
    if allowed_source_classes is not None and source_type not in allowed_source_classes:
        errors.append(f"{scene.scene_id}: source type should be in configured label space, got {source_type}")
    occlusion = label.get("occlusion", {})
    if occlusion.get("is_direct_path_obstructed") != label.get("relative", {}).get("is_obstructed"):
        errors.append(f"{scene.scene_id}: occlusion flag mismatch")
    rx = label.get("receiver", {}).get("center_xyz_m", [None, None, None])
    sx = label.get("source", {}).get("xyz_m", [None, None, None])
    if None not in rx[:2] and not scene.walkable_area.contains(Point(rx[0], rx[1])):
        errors.append(f"{scene.scene_id}: receiver outside walkable area")
    if None not in sx[:2] and not scene.walkable_area.contains(Point(sx[0], sx[1])):
        errors.append(f"{scene.scene_id}: source outside walkable area")
    wav = label.get("files", {}).get("audio_wav")
    wav_info = None
    if wav:
        wav_path = Path(wav)
        if not wav_path.exists():
            errors.append(f"{scene.scene_id}: wav missing")
        else:
            wav_info = sf.info(wav_path)
            if wav_info.channels != 4:
                errors.append(f"{scene.scene_id}: expected 4-channel FOA wav, got {wav_info.channels}")
            if wav_info.frames <= 0:
                errors.append(f"{scene.scene_id}: wav has no samples")
    mono_wav = label.get("files", {}).get("mono_wav")
    if mono_wav:
        mono_path = Path(mono_wav)
        if not mono_path.exists():
            errors.append(f"{scene.scene_id}: mono wav missing")
        else:
            mono_info = sf.info(mono_path)
            if mono_info.channels != 1:
                errors.append(f"{scene.scene_id}: expected 1-channel mono wav, got {mono_info.channels}")
            if mono_info.frames <= 0:
                errors.append(f"{scene.scene_id}: mono wav has no samples")
            if wav_info is not None and mono_info.frames != wav_info.frames:
                errors.append(f"{scene.scene_id}: mono wav frame count differs from FOA wav")
    return errors


def validate_obj(obj_path: Path) -> list[str]:
    errors: list[str] = []
    if not obj_path.exists():
        return [f"{obj_path}: missing OBJ"]
    text = obj_path.read_text(encoding="utf-8")
    for marker in ["g boundary_floor", "semantic_floor_material", "semantic_boundary_material", "semantic_ceiling_material"]:
        if marker not in text:
            errors.append(f"{obj_path.name}: missing marker {marker}")
    return errors


def summarize_validation(errors: list[str]) -> dict[str, Any]:
    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:100],
    }
