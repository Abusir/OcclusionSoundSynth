from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_geometric.occ_synth.acoustics import synthesize_dry_sound
from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes

from soundspaces_adapter.backend import SoundSpacesBackend, SoundSpacesUnavailableError, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.coordinate import assert_round_trip, occ_to_habitat
from soundspaces_adapter.materials import write_material_map
from soundspaces_adapter.validation import validate_rir_physics, write_validation_report
from soundspaces_adapter.visualize_debug import plot_geometry_debug, plot_rir_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and validate one SoundSpaces debug case.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/debug_one"))
    parser.add_argument("--scene-index", type=int, default=0, help="Index in the 60-scene generated catalog.")
    parser.add_argument("--case", choices=["los", "nlos"], default="nlos")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--ray-count", type=int, default=5000)
    parser.add_argument("--dry-source-type", type=str, default="fire")
    return parser.parse_args()


def find_placement(scene, rng: random.Random, want_los: bool):
    for _ in range(80):
        placement = sample_placement(
            scene,
            rng,
            source_types=["fire"],
            prefer_obstructed=not want_los,
        )
        if placement.is_los == want_los:
            return placement
    return placement


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    figure_dir = out / "figures"
    audio_dir = out / "audio"
    report_dir = out / "reports"
    for path in (geometry_dir, figure_dir, audio_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)
    scene = scenes[args.scene_index % len(scenes)]
    files = export_scene_obj(scene, geometry_dir)
    material_path = geometry_dir / f"{scene.scene_id}.sound_materials.json"
    material_map = write_material_map(material_path)

    rng = random.Random(args.seed + 17)
    placement = find_placement(scene, rng, want_los=args.case == "los")
    assert_round_trip(placement.receiver_xyz)
    assert_round_trip(placement.source_xyz)
    plot_geometry_debug(
        scene,
        placement,
        figure_dir / f"{args.case}_{scene.scene_id}_geometry.png",
        title=f"{scene.scene_id} / {args.case.upper()} geometry",
    )

    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        indirect_ray_count=args.ray_count,
        output_directory=str(out),
    )
    config.save_json(report_dir / "soundspaces_config.json")

    availability = check_soundspaces_available()
    base_report = {
        "scene_id": scene.scene_id,
        "scene_index": scene.scene_index,
        "case": args.case,
        "obj_path": files["obj"],
        "mtl_path": files["mtl"],
        "material_map_path": str(material_path),
        "material_map": material_map,
        "receiver_occ_xyz": placement.receiver_xyz,
        "source_occ_xyz": placement.source_xyz,
        "receiver_habitat_xyz": occ_to_habitat(placement.receiver_xyz).tolist(),
        "source_habitat_xyz": occ_to_habitat(placement.source_xyz).tolist(),
        "is_los": placement.is_los,
        "obstruction_count": placement.obstruction_count,
        "obstruction_types": placement.obstruction_types,
        "soundspaces_available": availability,
        "config": config.to_dict(),
    }

    if not availability["available"]:
        base_report["status"] = "soundspaces_unavailable"
        base_report["message"] = "Geometry, materials, coordinates, and plots were generated; install SoundSpaces/Habitat-Sim to render the real RIR."
        write_validation_report(report_dir / "render_one_report.json", base_report)
        print(json.dumps(base_report, indent=2))
        return 0

    try:
        backend = SoundSpacesBackend(config)
        rir = backend.render_rir(
            scene_mesh_path=Path(files["obj"]),
            source_occ_xyz=placement.source_xyz,
            receiver_occ_xyz=placement.receiver_xyz,
            output_dir=out,
        )
        np.save(report_dir / f"{args.case}_{scene.scene_id}_rir.npy", rir)
        plot_rir_debug(
            rir,
            placement.source_xyz,
            placement.receiver_xyz,
            config.sample_rate,
            figure_dir / f"{args.case}_{scene.scene_id}_rir.png",
            title=f"{scene.scene_id} / {args.case.upper()} SoundSpaces RIR",
        )
        validation = validate_rir_physics(
            rir,
            placement.source_xyz,
            placement.receiver_xyz,
            config.sample_rate,
            is_los=placement.is_los,
        )
        dry = synthesize_dry_sound(args.dry_source_type, config.sample_rate, args.duration, args.seed)
        audio_meta = backend.convolve_and_save(
            dry,
            rir,
            audio_dir / f"{args.case}_{scene.scene_id}_foa.wav",
            audio_dir / f"{args.case}_{scene.scene_id}_mono.wav",
        )
        base_report["status"] = "rendered"
        base_report["rir_shape"] = list(rir.shape)
        base_report["validation"] = validation.to_dict()
        base_report["audio"] = audio_meta
        write_validation_report(report_dir / "render_one_report.json", base_report)
        print(json.dumps(base_report, indent=2))
        return 0 if validation.passed else 2
    except SoundSpacesUnavailableError as exc:
        base_report["status"] = "soundspaces_api_mismatch"
        base_report["message"] = str(exc)
        write_validation_report(report_dir / "render_one_report.json", base_report)
        print(json.dumps(base_report, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
