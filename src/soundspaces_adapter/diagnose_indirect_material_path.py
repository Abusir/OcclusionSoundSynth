from __future__ import annotations

import argparse
import json
from pathlib import Path

from soundspaces_adapter.mechanism_verification import render_pair, write_material_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare indirect-only RIRs with and without semantic materials.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_soundspaces_runs/indirect_only_material_path_check"))
    parser.add_argument("--scene-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.5)
    parser.add_argument("--ray-count", type=int, default=50000)
    parser.add_argument("--source-ray-count", type=int, default=200)
    parser.add_argument("--source-ray-depth", type=int, default=10)
    parser.add_argument("--absorption", type=float, default=0.02)
    parser.add_argument("--scattering", type=float, default=0.50)
    parser.add_argument("--transmission", type=float, default=0.0)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--semantic-asset-kind", choices=["ply", "obj_colored"], default="obj_colored")
    parser.add_argument("--material-density", type=float, default=998.6546630859375)
    parser.add_argument("--material-speed", type=float, default=1483.9610595703125)
    parser.add_argument("--no-semantic-stage", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    material_json = args.output_dir / "low_absorption_default_labels.json"
    write_material_database(
        material_json,
        absorption=args.absorption,
        scattering=args.scattering,
        transmission=args.transmission,
        damping=args.damping,
        density=args.material_density,
        speed=args.material_speed,
    )
    report = render_pair(
        args.output_dir,
        scene_index=args.scene_index,
        want_los=True,
        seed=args.seed,
        sample_rate=args.sample_rate,
        ir_duration=args.ir_duration,
        config_a={
            "direct": False,
            "indirect": True,
            "diffraction": False,
            "transmission": False,
            "indirect_ray_count": args.ray_count,
            "source_ray_count": args.source_ray_count,
            "source_ray_depth": args.source_ray_depth,
            "enable_materials": False,
        },
        config_b={
            "direct": False,
            "indirect": True,
            "diffraction": False,
            "transmission": False,
            "indirect_ray_count": args.ray_count,
            "source_ray_count": args.source_ray_count,
            "source_ray_depth": args.source_ray_depth,
            "enable_materials": True,
            "audio_materials_json": str(material_json),
            "semantic_material_stage": not args.no_semantic_stage,
            "semantic_asset_kind": args.semantic_asset_kind,
        },
    )
    output = args.output_dir / "indirect_only_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "rendered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
