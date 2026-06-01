from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from _config import PROJECT_ROOT, SRC_ROOT, add_common_config_args, append_option, apply_overrides, load_yaml_config, resolve_path

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes
from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.visualize_debug import plot_geometry_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reproducible 3D scene assets for the browser viewer.")
    add_common_config_args(parser)
    return parser.parse_args()


def find_placement(scene, rng: random.Random, want_los: bool):
    placement = None
    for _ in range(160):
        placement = sample_placement(scene, rng, source_types=["example_source"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def viewer_relative(path: Path, viewer_root: Path) -> str:
    return path.resolve().relative_to(viewer_root.resolve()).as_posix()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def save_optional_habitat_rgb(cfg: dict[str, object], scene_mesh: Path, placement, output_dir: Path) -> dict[str, object]:
    if not bool(cfg.get("render_habitat_rgb", False)):
        return {}
    availability = check_soundspaces_available()
    if not availability.get("available"):
        return {"habitat_rgb": {"available": False, "reason": "soundspaces_unavailable", "detail": availability}}

    ss_cfg = SoundSpacesConfig(
        sample_rate=int(cfg.get("sample_rate", 16000)),
        ir_duration_s=float(cfg.get("ir_duration", 0.2)),
        direct_ray_count=int(cfg.get("direct_ray_count", 500)),
        indirect_ray_count=int(cfg.get("indirect_ray_count", cfg.get("ray_count", 1000))),
        enable_rgb=True,
        enable_depth=bool(cfg.get("render_depth", False)),
        visual_width=int(cfg.get("visual_width", 320)),
        visual_height=int(cfg.get("visual_height", 240)),
        visual_hfov_deg=float(cfg.get("visual_hfov_deg", 90.0)),
        visual_yaw_offset_deg=float(cfg.get("visual_yaw_offset_deg", 0.0)),
        visual_pitch_deg=float(cfg.get("visual_pitch_deg", 0.0)),
    )
    observations = SoundSpacesBackend(ss_cfg).render_visual_only(scene_mesh, placement.source_xyz, placement.receiver_xyz)
    result: dict[str, object] = {"habitat_rgb": {"available": True}}
    if "rgb_sensor" in observations:
        rgb = np.asarray(observations["rgb_sensor"])
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb_path = output_dir / "figures" / "habitat_rgb.png"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(rgb_path, rgb.astype(np.uint8))
        result["rgb"] = {"path": str(rgb_path), "shape": list(rgb.shape)}
    if "depth_sensor" in observations:
        depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
        depth_path = output_dir / "figures" / "habitat_depth.npy"
        np.save(depth_path, depth)
        result["depth"] = {"path": str(depth_path), "shape": list(depth.shape)}
    return result


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = apply_overrides(load_yaml_config(config_path), args.set)
    cfg_dir = config_path.parent

    output_dir = resolve_path(cfg.get("output_dir", "../outputs/visualization"), config_dir=cfg_dir)
    viewer_root = resolve_path(cfg.get("viewer_root", "../viewer3d"), config_dir=cfg_dir)
    viewer_data_dir = viewer_root / "data"
    geometry_dir = viewer_data_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = generate_all_scenes(variants_per_type=int(cfg.get("variants_per_type", 10)), seed=int(cfg.get("seed", 20260416)))
    scene_index = int(cfg.get("scene_index", 0))
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError(f"scene_index out of range: {scene_index}; catalog has {len(scenes)} scenes")
    scene = scenes[scene_index]
    want_los = str(cfg.get("case", "los")).lower() != "nlos"
    placement = find_placement(scene, random.Random(int(cfg.get("placement_seed", 20260416 + scene_index))), want_los)
    files = export_scene_obj(scene, geometry_dir)
    obj_path = Path(files["obj"]).resolve()

    layout_path = output_dir / "figures" / f"{scene.scene_id}_layout.png"
    viewer_figure_dir = viewer_data_dir / "figures"
    viewer_figure_dir.mkdir(parents=True, exist_ok=True)
    viewer_layout_path = viewer_figure_dir / layout_path.name
    camera_forward = None
    try:
        backend = SoundSpacesBackend(SoundSpacesConfig(visual_yaw_offset_deg=float(cfg.get("visual_yaw_offset_deg", 0.0))))
        camera_forward = backend.camera_forward_occ_xy(placement.source_xyz, placement.receiver_xyz)
    except Exception:
        camera_forward = None
    plot_geometry_debug(
        scene,
        placement,
        layout_path,
        title=f"{scene.scene_id} {cfg.get('case', 'los')}",
        show_camera=True,
        camera_hfov_deg=float(cfg.get("visual_hfov_deg", 90.0)),
        camera_forward_xy=camera_forward,
    )
    shutil.copy2(layout_path, viewer_layout_path)

    report = {
        "scene_id": scene.scene_id,
        "scene_index": scene.scene_index,
        "scene_type": scene.scene_type,
        "case": cfg.get("case", "los"),
        "source_xyz": list(placement.source_xyz),
        "receiver_xyz": list(placement.receiver_xyz),
        "is_los": bool(placement.is_los),
        "obstruction_count": int(placement.obstruction_count),
        "obstruction_types": list(placement.obstruction_types),
        "audio_visual_mesh": viewer_relative(obj_path, viewer_root),
        "layout_png": viewer_relative(viewer_layout_path, viewer_root),
        "config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
        "rgb_alignment": {
            "rgb_output_is_mirrored_horizontally": False,
            "camera_footprint_is_reversed_180_deg": False,
        },
    }
    report.update(save_optional_habitat_rgb(cfg, obj_path, placement, output_dir))

    report_path = output_dir / "scene_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_viewer_report = viewer_data_dir / "latest_report.json"
    latest_viewer_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "note": "Scene catalog generated from configs/visualization.yaml.",
        "entries": [report],
    }
    (viewer_data_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "viewer_report": str(latest_viewer_report), "layout_png": str(layout_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
