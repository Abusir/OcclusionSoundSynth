from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes

from soundspaces_adapter.backend import SoundSpacesBackend, SoundSpacesUnavailableError, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.visualize_debug import plot_geometry_debug
from soundspaces_adapter.verify_audio_visual import save_depth, save_rgb, write_reversed_winding_obj
from soundspaces_adapter.validation import write_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Habitat RGB/depth rendering without the audio sensor.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=3)
    parser.add_argument("--case", choices=["los", "nlos"], default="los")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--visual-pitch", type=float, default=0.0)
    parser.add_argument("--visual-yaw-offset", type=float, default=0.0)
    parser.add_argument(
        "--align-rgb-footprint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror RGB horizontally to match the top-down OCC left-right convention.",
    )
    return parser.parse_args()


def find_placement(scene, rng: random.Random, want_los: bool):
    placement = None
    for _ in range(120):
        placement = sample_placement(scene, rng, source_types=["fire"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    for path in (geometry_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    availability = check_soundspaces_available()
    report: dict[str, object] = {
        "soundspaces_available": availability,
        "mode": "visual_only",
        "checks": {
            "rgb": "RGB observation exists, has the requested resolution, and is nonblank.",
            "depth": "Depth observation is saved when available and has nontrivial variation.",
        },
    }
    if not availability["available"]:
        report["status"] = "soundspaces_unavailable"
        report["passed"] = False
        write_validation_report(report_dir / "visual_only_report.json", report)
        print(json.dumps(report, indent=2))
        return 3

    scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)
    scene = scenes[args.scene_index % len(scenes)]
    placement = find_placement(scene, random.Random(args.seed + 311), want_los=args.case == "los")
    files = export_scene_obj(scene, geometry_dir)
    render_mesh = write_reversed_winding_obj(Path(files["obj"]), geometry_dir / f"{scene.scene_id}_visual_interior.obj")

    plain_geometry_path = figure_dir / f"{args.case}_{scene.scene_id}_geometry.png"
    camera_geometry_path = figure_dir / f"{args.case}_{scene.scene_id}_geometry_camera.png"
    plot_geometry_debug(
        scene,
        placement,
        plain_geometry_path,
        title=f"{scene.scene_id} / {args.case.upper()} geometry",
        show_camera=False,
    )

    cfg = SoundSpacesConfig(
        output_directory=str(out),
        enable_rgb=True,
        enable_depth=True,
        visual_width=args.width,
        visual_height=args.height,
        visual_pitch_deg=args.visual_pitch,
        visual_yaw_offset_deg=args.visual_yaw_offset,
    )
    cfg.save_json(report_dir / "visual_only_config.json")
    try:
        backend = SoundSpacesBackend(cfg)
        camera_forward = backend.camera_forward_occ_xy(placement.source_xyz, placement.receiver_xyz)
        plot_geometry_debug(
            scene,
            placement,
            camera_geometry_path,
            title=f"{scene.scene_id} / {args.case.upper()} camera footprint",
            show_camera=True,
            camera_hfov_deg=cfg.visual_hfov_deg,
            camera_forward_xy=camera_forward,
        )
        observations = backend.render_visual_only(render_mesh, placement.source_xyz, placement.receiver_xyz)
    except SoundSpacesUnavailableError as exc:
        report["status"] = "soundspaces_api_mismatch"
        report["message"] = str(exc)
        report["passed"] = False
        write_validation_report(report_dir / "visual_only_report.json", report)
        print(json.dumps(report, indent=2))
        return 3

    raw_rgb_meta = save_rgb(figure_dir / f"{args.case}_{scene.scene_id}_rgb_raw.png", observations["rgb_sensor"])
    rgb_meta = save_rgb(
        figure_dir / f"{args.case}_{scene.scene_id}_rgb.png",
        observations["rgb_sensor"],
        mirror_horizontal=args.align_rgb_footprint,
    )
    depth_meta = save_depth(figure_dir / f"{args.case}_{scene.scene_id}_depth.png", observations["depth_sensor"])
    report.update(
        {
            "status": "rendered",
            "scene_id": scene.scene_id,
            "case": args.case,
            "is_los": placement.is_los,
            "source_xyz": placement.source_xyz,
            "receiver_xyz": placement.receiver_xyz,
            "audio_visual_mesh": str(render_mesh),
            "plain_geometry_figure": str(plain_geometry_path),
            "camera_geometry_figure": str(camera_geometry_path),
            "rgb_raw": raw_rgb_meta,
            "rgb_alignment": {
                "rgb_output_is_mirrored_horizontally": bool(args.align_rgb_footprint),
                "camera_footprint_is_reversed_180_deg": False,
                "note": (
                    "The aligned RGB output is mirrored horizontally to match the top-down OCC left-right convention. "
                    "The camera footprint uses the Habitat-facing forward direction directly. "
                    "The raw unmirrored image is saved beside the aligned RGB output."
                ),
            },
            "observations": sorted(observations.keys()),
            "rgb": rgb_meta,
            "depth": depth_meta,
        }
    )
    report["passed"] = bool(
        rgb_meta["shape"][:2] == [args.height, args.width]
        and rgb_meta["std"] > 5.0
        and rgb_meta["max"] >= 30
        and depth_meta["std"] > 5e-2
    )
    write_validation_report(report_dir / "visual_only_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
