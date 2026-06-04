from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import struct
from typing import Any

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

from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.semantic_stage import write_semantic_stage_for_obj
from soundspaces_adapter.validation import energy_envelope


def find_placement(scene, seed: int, want_los: bool):
    rng = random.Random(seed)
    for _ in range(160):
        placement = sample_placement(scene, rng, source_types=["fire"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def energy_summary(rir: np.ndarray, sample_rate: int) -> dict[str, object]:
    env = energy_envelope(rir)
    energy = env * env
    early_stop = min(env.shape[0], int(round(sample_rate * 0.05)))
    return {
        "shape": list(rir.shape),
        "total_energy": float(np.sum(energy)),
        "early_50ms_energy": float(np.sum(energy[:early_stop])),
        "late_after_50ms_energy": float(np.sum(energy[early_stop:])),
        "peak_sample": int(np.argmax(env)) if env.size else None,
        "peak_value": float(np.max(env)) if env.size else 0.0,
        "nonzero_samples": int(np.count_nonzero(env > 0.0)),
    }


def normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    ea = energy_envelope(a)
    eb = energy_envelope(b)
    n = min(ea.shape[0], eb.shape[0])
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(ea[:n] - eb[:n]) / (np.linalg.norm(eb[:n]) + 1e-12))


def ratio(a: float, b: float) -> float:
    return float(a / (b + 1e-12))


def write_material_database(
    path: Path,
    *,
    absorption: float,
    scattering: float,
    transmission: float,
    damping: float | None = None,
    density: float = 998.6546630859375,
    speed: float = 1483.9610595703125,
) -> None:
    frequencies = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    damping_freqs = [
        22.27948,
        27.64745,
        34.30876,
        42.57504,
        52.83297,
        65.56244,
        81.35888,
        100.96133,
        125.28674,
        155.47299,
        192.93234,
        239.41707,
        297.10159,
        368.68466,
        457.51477,
        567.74719,
        704.53906,
        874.28882,
        1084.93823,
        1346.34106,
        1670.72485,
        2073.26587,
        2572.79443,
        3192.67676,
        3961.91577,
        4916.48926,
        6101.05518,
        7571.03467,
        9395.17969,
        11658.83008,
        14467.89160,
        17953.74805,
    ]

    def curve(value: float) -> list[float]:
        out: list[float] = []
        for freq in frequencies:
            out.extend([freq, value])
        return out

    def damping_curve() -> list[float]:
        out: list[float] = []
        base = 1.1595274e-10
        for idx, freq in enumerate(damping_freqs):
            value = float(damping) if damping is not None else base * (1.54 ** idx)
            out.extend([freq, value])
        return out

    label_aliases = {
        "default": ["default"],
        "floor": ["floor", "indoor_floor_hard", "outdoor_ground_grass"],
        "wall": ["wall", "indoor_wall_reflective", "outdoor_wall_reflective"],
        "ceiling": ["ceiling", "indoor_ceiling_reflective", "outdoor_ceiling", "sky_absorber"],
        "obstacle": ["obstacle", "solid_occluder"],
    }
    materials = []
    for label, aliases in label_aliases.items():
        materials.append(
            {
                "name": f"{label}_{absorption:.2f}_{transmission:.2f}",
                "labels": aliases,
                "absorption": curve(absorption),
                "scattering": curve(scattering),
                "transmission": curve(transmission),
                "damping": damping_curve(),
                "density": float(density),
                "speed": float(speed),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"materials": materials}, indent=2), encoding="utf-8")


def write_reversed_winding_obj(src: Path, dst: Path) -> Path:
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
    if mtl_src.exists():
        mtl_dst.write_text(mtl_src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def write_minimal_semantic_stage(obj_path: Path) -> Path:
    """Write a Habitat stage config that makes generated OBJ usable as semantic mesh.

    Generated OCC scenes do not yet have per-face semantic annotations. This
    minimal config intentionally maps the semantic asset to the same mesh and
    provides broad category labels so SoundSpaces can exercise its material
    database path without crashing on a missing semantic resource.
    """

    scene_id = obj_path.stem
    scn_path = obj_path.with_suffix(".scn")
    ply_path = obj_path.with_name(f"{scene_id}_semantic.ply")
    stage_path = obj_path.with_name(f"{scene_id}.stage_config.json")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            raw = [part.split("/")[0] for part in line.split()[1:]]
            idx = [int(item) - 1 for item in raw]
            if len(idx) == 3:
                faces.append((idx[0], idx[1], idx[2]))
            elif len(idx) > 3:
                for offset in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[offset], idx[offset + 1]))
    ply_header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
        ]
    ) + "\n"
    with ply_path.open("wb") as handle:
        handle.write(ply_header.encode("ascii"))
        for x, y, z in vertices:
            handle.write(struct.pack("<fffBBB", x, y, z, 0, 0, 0))
        for a, b, c in faces:
            handle.write(struct.pack("<Biii", 3, a, b, c))
    objects = [
        {"class_": "default", "id": 0},
        {"class_": "floor", "id": 1},
        {"class_": "wall", "id": 2},
        {"class_": "ceiling", "id": 3},
        {"class_": "obstacle", "id": 4},
    ]
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


def render_pair(
    out: Path,
    *,
    scene_index: int,
    want_los: bool,
    seed: int,
    sample_rate: int,
    ir_duration: float,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    availability = check_soundspaces_available()
    if not availability.get("available"):
        return {"status": "soundspaces_unavailable", "soundspaces_available": availability, "passed": False}
    scenes = generate_all_scenes(variants_per_type=10, seed=20260416)
    scene = scenes[scene_index % len(scenes)]
    placement = find_placement(scene, seed + 701, want_los=want_los)
    geometry_dir = out / "geometry"
    files = export_scene_obj(scene, geometry_dir)
    base_mesh = Path(files["obj"])
    use_semantic_material_stage = any(
        bool(overrides.get("enable_materials") or overrides.get("audio_materials_json"))
        and bool(overrides.get("semantic_material_stage", True))
        for overrides in (config_a, config_b)
    )
    semantic_asset_kind = str(config_b.get("semantic_asset_kind", config_a.get("semantic_asset_kind", "ply")))
    render_mesh = (
        write_semantic_stage_for_obj(base_mesh, semantic_asset_kind=semantic_asset_kind)
        if use_semantic_material_stage
        else base_mesh
    )
    rirs: dict[str, np.ndarray] = {}
    summaries: dict[str, object] = {}
    for name, overrides in [("a", config_a), ("b", config_b)]:
        cfg_kwargs = {
            "sample_rate": sample_rate,
            "ir_duration_s": ir_duration,
            "output_directory": str(out / name),
            "thread_count": 1,
            "enable_rgb": False,
            "enable_depth": False,
        }
        cfg_kwargs.update(overrides)
        cfg = SoundSpacesConfig(**cfg_kwargs)
        backend = SoundSpacesBackend(cfg)
        rir = np.asarray(backend.render_rir(render_mesh, placement.source_xyz, placement.receiver_xyz, out / name), dtype=np.float32)
        rirs[name] = rir
        np.save(out / f"{name}_rir.npy", rir)
        summaries[name] = energy_summary(rir, sample_rate)
    l2 = normalized_l2(rirs["a"], rirs["b"])
    sa = summaries["a"]
    sb = summaries["b"]
    return {
        "status": "rendered",
        "scene_id": scene.scene_id,
        "scene_index": scene.scene_index,
        "mesh_path": str(render_mesh),
        "is_los": placement.is_los,
        "source_xyz": placement.source_xyz,
        "receiver_xyz": placement.receiver_xyz,
        "obstruction_types": placement.obstruction_types,
        "summaries": summaries,
        "normalized_l2_a_to_b": l2,
        "total_energy_ratio_a_to_b": ratio(float(sa["total_energy"]), float(sb["total_energy"])),
    }


def run_worker(script_path: Path, args: list[str], output_path: Path, timeout_s: int = 180) -> dict[str, object]:
    cmd = [sys.executable, str(script_path), "--worker", *args]
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/occ_mpl"
    current_pythonpath = env.get("PYTHONPATH", "")
    needed_pythonpath = f"{ROOT / 'src'}:{ROOT}"
    env["PYTHONPATH"] = f"{needed_pythonpath}:{current_pythonpath}" if current_pythonpath else needed_pythonpath
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s, env=env)
    report: dict[str, object] = {
        "worker_command": cmd,
        "worker_returncode": proc.returncode,
        "worker_stdout_tail": proc.stdout[-4000:],
        "worker_stderr_tail": proc.stderr[-4000:],
    }
    if output_path.exists():
        try:
            report["worker_report"] = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report["worker_report_parse_error"] = str(exc)
    if proc.returncode < 0:
        report["native_crash_signal"] = -proc.returncode
    return report
