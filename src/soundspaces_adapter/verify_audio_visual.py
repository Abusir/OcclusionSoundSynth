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
from soundspaces_adapter.material_database import write_occ_material_database
from soundspaces_adapter.validation import validate_rir_physics, write_validation_report
from soundspaces_adapter.visualize_debug import plot_geometry_debug, plot_rir_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify joint audio-visual Habitat-Sim/SoundSpaces rendering.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audio_visual_verification"))
    parser.add_argument("--scene-index", type=int, default=3)
    parser.add_argument("--case", choices=["los", "nlos"], default="los")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--ray-count", type=int, default=1000)
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
    for _ in range(120):
        placement = sample_placement(scene, rng, source_types=["fire"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def write_reversed_winding_obj(src: Path, dst: Path) -> Path:
    """Write a diagnostic OBJ with identical vertices and reversed face winding.

    Generated room meshes are acoustically valid, but some faces are invisible
    from the camera side in Habitat's rasterizer. Reversing winding is useful for
    visual diagnostics because it leaves the metric geometry unchanged while
    making interior surfaces visible to RGB/depth sensors.
    """

    out_lines: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("mtllib "):
            out_lines.append(line.replace(src.with_suffix(".mtl").name, dst.with_suffix(".mtl").name))
        elif line.startswith("f "):
            parts = line.split()
            out_lines.append(" ".join([parts[0], *reversed(parts[1:])]))
        else:
            out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    mtl_src = src.with_suffix(".mtl")
    mtl_dst = dst.with_suffix(".mtl")
    if mtl_src.exists() and not mtl_dst.exists():
        mtl_dst.write_text(mtl_src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def save_rgb(path: Path, rgb: np.ndarray, *, mirror_horizontal: bool = False) -> dict[str, object]:
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if mirror_horizontal:
        arr = arr[:, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "min": int(arr.min()) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def save_depth(path: Path, depth: np.ndarray) -> dict[str, object]:
    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    valid = arr[finite]
    if valid.size:
        upper = float(np.percentile(valid, 99.0))
        lower = float(np.percentile(valid, 1.0))
        denom = max(upper - lower, 1e-6)
        norm = np.clip((arr - lower) / denom, 0.0, 1.0)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
    image = (norm * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "finite": bool(np.all(finite)),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "std": float(valid.std()) if valid.size else 0.0,
    }


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
        "mode": "joint_audio_visual",
        "checks": {
            "audio": "RIR is nonzero and passes the physics validator.",
            "rgb": "RGB observation exists, has the requested resolution, and is nonblank.",
            "depth": "Depth observation is saved when available; this old Habitat-Sim audio branch may return zero depth for programmatic OBJ scenes.",
        },
    }
    if not availability["available"]:
        report["status"] = "soundspaces_unavailable"
        report["passed"] = False
        write_validation_report(report_dir / "audio_visual_report.json", report)
        print(json.dumps(report, indent=2))
        return 3

    scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)
    scene = scenes[args.scene_index % len(scenes)]
    placement = find_placement(scene, random.Random(args.seed + 311), want_los=args.case == "los")
    files = export_scene_obj(scene, geometry_dir)
    render_mesh = write_reversed_winding_obj(Path(files["obj"]), geometry_dir / f"{scene.scene_id}_visual_interior.obj")
    material_db_path = report_dir / "occ_rlr_materials.json"
    write_occ_material_database(material_db_path)
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
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        indirect_ray_count=args.ray_count,
        output_directory=str(out),
        enable_rgb=True,
        enable_depth=True,
        visual_width=args.width,
        visual_height=args.height,
        visual_pitch_deg=args.visual_pitch,
        visual_yaw_offset_deg=args.visual_yaw_offset,
        audio_materials_json=str(material_db_path),
    )
    cfg.save_json(report_dir / "soundspaces_av_config.json")
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
        observations = backend.render_audio_visual(render_mesh, placement.source_xyz, placement.receiver_xyz, out)
    except SoundSpacesUnavailableError as exc:
        report["status"] = "soundspaces_api_mismatch"
        report["message"] = str(exc)
        report["passed"] = False
        write_validation_report(report_dir / "audio_visual_report.json", report)
        print(json.dumps(report, indent=2))
        return 3

    rir = np.asarray(observations.get("audio_sensor"), dtype=np.float32)
    np.save(report_dir / f"{args.case}_{scene.scene_id}_rir.npy", rir)
    plot_rir_debug(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, figure_dir / f"{args.case}_{scene.scene_id}_rir.png")
    validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, placement.is_los)
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
            "material_database": str(material_db_path),
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
            "angle_reference": {
                "zero_direction": "+x axis",
                "positive_direction": "counter_clockwise toward +y",
                "plane": "xy",
            },
            "audio_visual_mesh_note": (
                "This diagnostic mesh reverses face winding to make interior programmatic OBJ surfaces visible "
                "to Habitat RGB/depth sensors. Vertex positions and metric geometry are unchanged."
            ),
            "semantic_material_note": (
                "The RLR material database is written and connected. Habitat-Sim will still use default material "
                "until generated scenes also provide a semantic mesh/semantic scene descriptor."
            ),
            "observations": sorted(observations.keys()),
            "rir_shape": list(rir.shape),
            "validation": validation.to_dict(),
            "rgb": rgb_meta,
            "depth": depth_meta,
        }
    )
    report["passed"] = bool(
        validation.passed
        and np.any(np.abs(rir) > 0.0)
        and rgb_meta["shape"][:2] == [args.height, args.width]
        and rgb_meta["std"] > 5.0
        and rgb_meta["max"] >= 30
        and depth_meta["std"] > 5e-2
    )
    write_validation_report(report_dir / "audio_visual_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
