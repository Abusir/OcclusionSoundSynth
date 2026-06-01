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

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes

from soundspaces_adapter.backend import SoundSpacesBackend, SoundSpacesUnavailableError, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.materials import write_material_map
from soundspaces_adapter.validation import energy_envelope, validate_rir_physics, write_validation_report
from soundspaces_adapter.visualize_debug import plot_geometry_debug, plot_rir_comparison, plot_rir_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SoundSpaces RIR physical validation suite.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physics_suite"))
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--case", choices=["los", "nlos"], default="nlos")
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=1.0)
    parser.add_argument("--ray-counts", type=str, default="1000,5000,20000")
    return parser.parse_args()


def find_placement(scene, rng: random.Random, want_los: bool):
    for _ in range(100):
        placement = sample_placement(scene, rng, source_types=["fire"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    ea = energy_envelope(a)
    eb = energy_envelope(b)
    n = min(ea.shape[0], eb.shape[0])
    if n == 0:
        return float("nan")
    ea = ea[:n]
    eb = eb[:n]
    denom = float(np.linalg.norm(eb) + 1e-12)
    return float(np.linalg.norm(ea - eb) / denom)


def rir_energy_summary(rir: np.ndarray, sample_rate: int) -> dict[str, object]:
    envelope = energy_envelope(rir)
    if envelope.size == 0:
        return {
            "sample_count": 0,
            "peak_sample": None,
            "peak_value": 0.0,
            "total_energy": 0.0,
            "early_50ms_energy": 0.0,
            "late_after_50ms_energy": 0.0,
            "nonzero_sample_count": 0,
        }
    early_stop = min(envelope.shape[0], int(round(sample_rate * 0.05)))
    energy = envelope * envelope
    return {
        "sample_count": int(envelope.shape[0]),
        "peak_sample": int(np.argmax(envelope)),
        "peak_value": float(np.max(envelope)),
        "total_energy": float(np.sum(energy)),
        "early_50ms_energy": float(np.sum(energy[:early_stop])),
        "late_after_50ms_energy": float(np.sum(energy[early_stop:])),
        "nonzero_sample_count": int(np.count_nonzero(envelope > 0.0)),
    }


def energy_ratio(a: np.ndarray, reference: np.ndarray, sample_rate: int) -> dict[str, float]:
    ea = rir_energy_summary(a, sample_rate)
    eb = rir_energy_summary(reference, sample_rate)
    return {
        "total_energy_ratio_to_full": float(ea["total_energy"]) / (float(eb["total_energy"]) + 1e-12),
        "early_50ms_energy_ratio_to_full": float(ea["early_50ms_energy"]) / (float(eb["early_50ms_energy"]) + 1e-12),
        "late_after_50ms_energy_ratio_to_full": float(ea["late_after_50ms_energy"])
        / (float(eb["late_after_50ms_energy"]) + 1e-12),
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    for path in (geometry_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)
    scene = scenes[args.scene_index % len(scenes)]
    placement = find_placement(scene, random.Random(args.seed + 177), want_los=args.case == "los")
    files = export_scene_obj(scene, geometry_dir)
    write_material_map(geometry_dir / f"{scene.scene_id}.sound_materials.json")
    plot_geometry_debug(scene, placement, figure_dir / f"{args.case}_{scene.scene_id}_geometry.png")

    ray_counts = [int(part) for part in args.ray_counts.split(",") if part.strip()]
    availability = check_soundspaces_available()
    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "case": args.case,
        "is_los": placement.is_los,
        "obstruction_types": placement.obstruction_types,
        "obj_path": files["obj"],
        "ray_counts": ray_counts,
        "soundspaces_available": availability,
        "tests": {
            "direct_delay": "LOS first peak should match distance / 343 * sample_rate.",
            "ray_convergence": "Increasing ray count should reduce envelope differences.",
            "ablation": "Disabling propagation mechanisms should change physically relevant RIR energy.",
        },
        "pass_criteria": {
            "los_direct_delay": "Every LOS validation must pass validate_rir_physics.",
            "nonzero_primary_render": "Ray-count renders and full ablation render must contain nonzero energy.",
            "ablation_sensitivity": "At least one non-full ablation must differ from full by normalized envelope L2 > 1e-4.",
        },
    }

    if not availability["available"]:
        report["status"] = "soundspaces_unavailable"
        report["message"] = "Install SoundSpaces/Habitat-Sim to run ray convergence and ablation rendering."
        write_validation_report(report_dir / "physics_suite_report.json", report)
        print(json.dumps(report, indent=2))
        return 0

    rendered: dict[str, np.ndarray] = {}
    validations: dict[str, object] = {}
    try:
        for ray_count in ray_counts:
            cfg = SoundSpacesConfig(
                sample_rate=args.sample_rate,
                ir_duration_s=args.ir_duration,
                indirect_ray_count=ray_count,
                output_directory=str(out),
            )
            backend = SoundSpacesBackend(cfg)
            name = f"rays_{ray_count}"
            rir = backend.render_rir(Path(files["obj"]), placement.source_xyz, placement.receiver_xyz, out)
            rendered[name] = rir
            np.save(report_dir / f"{name}_rir.npy", rir)
            validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, cfg.sample_rate, placement.is_los)
            validations[name] = validation.to_dict()
            validations[name]["energy_summary"] = rir_energy_summary(rir, cfg.sample_rate)
            plot_rir_debug(rir, placement.source_xyz, placement.receiver_xyz, cfg.sample_rate, figure_dir / f"{name}_rir.png")

        full_cfg = SoundSpacesConfig(sample_rate=args.sample_rate, ir_duration_s=args.ir_duration, output_directory=str(out))
        ablations = {
            "full": {},
            "no_indirect": {"indirect": False},
            "no_diffraction": {"diffraction": False},
            "no_transmission": {"transmission": False},
        }
        for name, overrides in ablations.items():
            cfg_kwargs = full_cfg.to_dict()
            cfg_kwargs.update(overrides)
            cfg = SoundSpacesConfig(**cfg_kwargs)
            backend = SoundSpacesBackend(cfg)
            rir = backend.render_rir(Path(files["obj"]), placement.source_xyz, placement.receiver_xyz, out)
            rendered[f"ablation_{name}"] = rir
            np.save(report_dir / f"ablation_{name}_rir.npy", rir)

        if rendered:
            plot_rir_comparison(rendered, args.sample_rate, figure_dir / "rir_comparison.png")
        convergence = {}
        if ray_counts:
            reference = rendered.get(f"rays_{ray_counts[-1]}")
            if reference is not None:
                for ray_count in ray_counts[:-1]:
                    key = f"rays_{ray_count}"
                    if key in rendered:
                        convergence[key] = normalized_l2(rendered[key], reference)
        ablation_metrics = {}
        full = rendered.get("ablation_full")
        if full is not None:
            for name in ("no_indirect", "no_diffraction", "no_transmission"):
                key = f"ablation_{name}"
                if key in rendered:
                    ablation_metrics[name] = {
                        "normalized_l2_to_full": normalized_l2(rendered[key], full),
                        **energy_ratio(rendered[key], full, args.sample_rate),
                        "energy_summary": rir_energy_summary(rendered[key], args.sample_rate),
                    }
            ablation_metrics["full"] = {"energy_summary": rir_energy_summary(full, args.sample_rate)}
        validation_passes = [bool(item["passed"]) for item in validations.values()]
        primary_nonzero_keys = [f"rays_{ray_count}" for ray_count in ray_counts]
        primary_nonzero_keys.append("ablation_full")
        nonzero_passes = [
            bool(rir_energy_summary(rendered[key], args.sample_rate)["total_energy"] > 0.0)
            for key in primary_nonzero_keys
            if key in rendered
        ]
        ablation_diffs = [
            float(metrics["normalized_l2_to_full"])
            for name, metrics in ablation_metrics.items()
            if name != "full" and "normalized_l2_to_full" in metrics
        ]
        report_passed = bool(
            (all(validation_passes) if validation_passes else False)
            and (all(nonzero_passes) if nonzero_passes else False)
            and (max(ablation_diffs) > 1e-4 if ablation_diffs else False)
        )
        report["status"] = "rendered"
        report["validations"] = validations
        report["ray_convergence_normalized_l2_to_max_ray_count"] = convergence
        report["ablation_metrics"] = ablation_metrics
        report["passed"] = report_passed
        write_validation_report(report_dir / "physics_suite_report.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report_passed else 2
    except SoundSpacesUnavailableError as exc:
        report["status"] = "soundspaces_api_mismatch"
        report["message"] = str(exc)
        write_validation_report(report_dir / "physics_suite_report.json", report)
        print(json.dumps(report, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
