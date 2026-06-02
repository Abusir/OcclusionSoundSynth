from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes
from soundspaces_adapter.backend import SoundSpacesBackend
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.material_database import scene_material_assignment, write_occ_material_database
from soundspaces_adapter.semantic_stage import write_semantic_stage_for_obj


def _usemtl_values(obj_path: Path) -> list[str]:
    values: list[str] = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("usemtl "):
            material = line.split(maxsplit=1)[1].strip()
            if material not in values:
                values.append(material)
    return values


def _semantic_classes(scn_path: Path) -> list[str]:
    payload = json.loads(scn_path.read_text(encoding="utf-8"))
    return [str(item["class_"]) for item in payload.get("objects", [])]


def verify_scene(scene_type: str, output_dir: Path, render: bool, sample_rate: int, ir_duration: float) -> dict[str, object]:
    scenes = [scene for scene in generate_all_scenes(variants_per_type=1) if scene.scene_type == scene_type]
    if not scenes:
        raise ValueError(f"unknown scene type: {scene_type}")
    scene = scenes[0]
    scene_dir = output_dir / scene_type
    geometry_dir = scene_dir / "geometry"
    report_dir = scene_dir / "reports"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = export_scene_obj(scene, geometry_dir)
    obj_path = Path(files["obj"])
    stage_path = write_semantic_stage_for_obj(obj_path)
    scn_path = obj_path.with_suffix(".scn")
    ply_path = obj_path.with_name(f"{obj_path.stem}_semantic.ply")
    materials_json = report_dir / "occ_rlr_materials.json"
    write_occ_material_database(materials_json)

    usemtl = _usemtl_values(obj_path)
    required = {"sky_absorber"}
    assignment = scene_material_assignment(scene.scene_type)
    for key in ("floor", "obstacle"):
        value = assignment.get(key)
        if value:
            required.add(value)
    missing = sorted(required.difference(usemtl))

    result: dict[str, object] = {
        "scene_type": scene.scene_type,
        "obj": str(obj_path),
        "stage_config": str(stage_path),
        "semantic_ply": str(ply_path),
        "semantic_scn": str(scn_path),
        "materials_json": str(materials_json),
        "usemtl": usemtl,
        "required_materials": sorted(required),
        "missing_materials": missing,
        "semantic_classes": _semantic_classes(scn_path),
        "semantic_stage_exists": stage_path.exists() and ply_path.exists() and scn_path.exists(),
    }

    if render:
        placement = sample_placement(scene, random.Random(20260416), source_types=["verify_open_boundary"])
        config = SoundSpacesConfig(
            sample_rate=sample_rate,
            ir_duration_s=ir_duration,
            direct_ray_count=100,
            indirect_ray_count=100,
            output_directory=str(scene_dir),
            enable_materials=True,
            audio_materials_json=str(materials_json),
        )
        rir = SoundSpacesBackend(config).render_rir(obj_path, placement.source_xyz, placement.receiver_xyz, scene_dir)
        rir_array = rir.foa if hasattr(rir, "foa") else rir
        result["render"] = {
            "rir_shape": list(rir_array.shape),
            "is_los": bool(placement.is_los),
            "note": "Check process logs for AudioSensor::setAudioMaterialsJSON and Loading semantic scene.",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify outdoor open-boundary absorber geometry and semantic materials.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_soundspaces_runs/open_boundary_verification"))
    parser.add_argument("--render", action="store_true", help="Run one SoundSpaces render per outdoor scene.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    reports = [
        verify_scene("open_field", args.output_dir, args.render, args.sample_rate, args.ir_duration),
        verify_scene("obstacle_forest", args.output_dir, args.render, args.sample_rate, args.ir_duration),
    ]
    failures = [report for report in reports if report["missing_materials"] or not report["semantic_stage_exists"]]
    print(json.dumps({"reports": reports, "passed": not failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
