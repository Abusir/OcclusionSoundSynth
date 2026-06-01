from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rir_bank.manifest import write_manifest_csv, write_manifest_jsonl
from rir_bank.metrics import compute_rir_metrics, summarize_metric_rows, validate_mono_rir
from rir_bank.validation import EXPECTED_FOA_ORDER, EXPECTED_MONO_DERIVATION, validate_manifest_rows


FOA_NORMALIZATION = "SN3D"
RIR_FORMAT = "both"
METRIC_CHANNEL = "mono_from_w_sqrt2_aligned"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an OCC SoundSpaces RIR-only bank.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=60, help="Number of programmatic scenes to use.")
    parser.add_argument("--rirs-per-scenario", type=int, default=None)
    parser.add_argument("--num-rirs", type=int, default=None)
    parser.add_argument("--scene-sampling", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--ray-count", type=int, default=1000, help="Backward-compatible alias for indirect ray count.")
    parser.add_argument("--direct-ray-count", type=int, default=500)
    parser.add_argument("--indirect-ray-count", type=int, default=1000)
    parser.add_argument("--preserve-propagation-delay", action="store_true", help="Disable SoundSpaces-style first-arrival onset alignment.")
    parser.add_argument("--onset-threshold-db", type=float, default=-80.0)
    parser.add_argument("--mono-scale", type=float, default=float(np.sqrt(2.0)), help="Match convolve_and_save mono = FOA_W * sqrt(2).")
    parser.add_argument("--rir-format", choices=["both", "foa"], default="both")
    parser.add_argument("--mono-from-foa-w", action="store_true", default=True)
    parser.add_argument("--compute-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def rir_onset_sample(rir_tc: np.ndarray, threshold_db: float = -80.0) -> int:
    """Match SoundSpacesBackend._rir_onset_sample for [T, C] RIR arrays."""

    arr = np.asarray(rir_tc, dtype=np.float32)
    if arr.ndim == 1:
        envelope = np.abs(arr)
    else:
        envelope = np.max(np.abs(arr), axis=1)
    if envelope.size == 0:
        return 0
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return 0
    threshold = peak * (10.0 ** (float(threshold_db) / 20.0))
    hits = np.flatnonzero(envelope >= threshold)
    return int(hits[0]) if hits.size else 0


def normalize_foa_rir(
    raw_rir: np.ndarray,
    expected_samples: int,
    align_output_onset: bool = True,
    onset_threshold_db: float = -80.0,
) -> tuple[np.ndarray, int]:
    """Return FOA as [4, T] using the same alignment/channel convention as convolve_and_save()."""

    arr = np.asarray(raw_rir, dtype=np.float32)
    if arr.ndim == 1:
        raise ValueError("SoundSpaces returned mono RIR; FOA Ambisonics RIR is required")
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D RIR, got shape {list(arr.shape)}")
    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 4:
        raise ValueError(f"expected at least 4 FOA channels, got shape {list(arr.shape)}")
    arr = arr[:, :4]

    onset = rir_onset_sample(arr, onset_threshold_db) if align_output_onset else 0
    if align_output_onset and onset > 0:
        arr = arr[onset:]

    # Match SoundSpacesBackend.convolve_and_save(): SoundSpaces/Habitat returns
    # directional channels in Habitat coordinates; project exports OCC-centric
    # ACN/SN3D [W,Y,Z,X], so Y/Z are swapped before saving.
    arr = arr[:, [0, 2, 1, 3]]
    if arr.shape[0] < expected_samples:
        pad = np.zeros((expected_samples - arr.shape[0], 4), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > expected_samples:
        arr = arr[:expected_samples]
    return np.ascontiguousarray(arr.T.astype(np.float32)), int(onset)


def write_config_yaml(path: Path, args: argparse.Namespace, config: SoundSpacesConfig, availability: dict[str, object]) -> None:
    payload = {
        "output_dir": str(args.output_dir),
        "scenarios": args.scenarios,
        "rirs_per_scenario": args.rirs_per_scenario,
        "num_rirs": args.num_rirs,
        "scene_sampling": args.scene_sampling,
        "sample_rate": args.sample_rate,
        "ir_duration": args.ir_duration,
        "ray_count": args.indirect_ray_count,
        "direct_ray_count": args.direct_ray_count,
        "indirect_ray_count": args.indirect_ray_count,
        "align_output_onset": not args.preserve_propagation_delay,
        "onset_threshold_db": args.onset_threshold_db,
        "mono_scale": args.mono_scale,
        "rir_format": args.rir_format,
        "mono_derivation": EXPECTED_MONO_DERIVATION,
        "foa_channel_order": EXPECTED_FOA_ORDER,
        "foa_normalization": FOA_NORMALIZATION,
        "metric_channel": METRIC_CHANNEL,
        "seed": args.seed,
        "soundspaces": availability,
        "soundspaces_config": config.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_to_simple_yaml(payload), encoding="utf-8")


def _to_simple_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines) + ("\n" if indent == 0 else "")
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines)
    return f"{prefix}{json.dumps(value, ensure_ascii=False)}"


def total_targets(args: argparse.Namespace, scene_count: int) -> list[int]:
    if args.rirs_per_scenario is not None:
        if args.rirs_per_scenario <= 0:
            raise ValueError("--rirs-per-scenario must be positive")
        return [args.rirs_per_scenario for _ in range(scene_count)]
    if args.num_rirs is None:
        raise ValueError("Provide --rirs-per-scenario or --num-rirs")
    if args.num_rirs <= 0:
        raise ValueError("--num-rirs must be positive")
    if args.scene_sampling == "stratified":
        base = args.num_rirs // scene_count
        rem = args.num_rirs % scene_count
        return [base + (1 if i < rem else 0) for i in range(scene_count)]
    targets = [0 for _ in range(scene_count)]
    rng = random.Random(args.seed + 19)
    for _ in range(args.num_rirs):
        targets[rng.randrange(scene_count)] += 1
    return targets


def occlusion_type(placement: Any) -> str:
    if placement.is_los:
        return "none"
    if placement.obstruction_types:
        return "|".join(placement.obstruction_types)
    return "unknown_occlusion"


def simulator_version(availability: dict[str, object]) -> str:
    module = availability.get("habitat_sim_module")
    return f"soundspaces2/habitat_sim:{module}" if module else "soundspaces2/habitat_sim"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from legacy_geometric.occ_synth.extrusion import export_scene_obj
    from legacy_geometric.occ_synth.sampling import sample_placement
    from legacy_geometric.occ_synth.scene_generator import generate_all_scenes
    from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
    from soundspaces_adapter.config import SoundSpacesConfig

    if args.scenarios <= 0:
        raise ValueError("--scenarios must be positive")
    if args.scenarios > 60:
        raise ValueError("The current programmatic catalog contains 60 scenes; use --scenarios <= 60")
    if args.rir_format == "both" and not args.mono_from_foa_w:
        raise ValueError("The first RIR bank version only supports mono derivation from FOA W channel.")

    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    rirs_dir = out / "rirs"
    reports_dir = out / "reports"
    cases_dir = out / "cases"
    for path in (geometry_dir, rirs_dir, reports_dir, cases_dir):
        path.mkdir(parents=True, exist_ok=True)

    availability = check_soundspaces_available()
    if not availability.get("available"):
        report = {"passed": False, "reason": "soundspaces_unavailable", "detail": availability}
        (reports_dir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        direct_ray_count=args.direct_ray_count,
        indirect_ray_count=args.indirect_ray_count,
        align_output_onset=not args.preserve_propagation_delay,
        onset_threshold_db=args.onset_threshold_db,
        output_directory=str(out),
        channel_type="Ambisonics",
        channel_count=4,
        enable_materials=False,
        enable_rgb=False,
        enable_depth=False,
    )
    write_config_yaml(out / "config.yaml", args, config, availability)

    scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)[: args.scenarios]
    for scene in scenes:
        export_scene_obj(scene, geometry_dir)

    backend = SoundSpacesBackend(config)
    expected_samples = int(round(args.sample_rate * args.ir_duration))
    targets = total_targets(args, len(scenes))
    rows: list[dict[str, Any]] = []
    progress_iter = range(sum(targets))
    progress = tqdm(progress_iter, desc="RIR bank", unit="rir", dynamic_ncols=True) if tqdm is not None else progress_iter
    progress_index = 0

    for scene_index, (scene, count) in enumerate(zip(scenes, targets)):
        scene_rir_dir = rirs_dir / f"scene_{scene_index:03d}"
        scene_rir_dir.mkdir(parents=True, exist_ok=True)
        for local_index in range(count):
            rir_index = len(rows)
            rir_id = f"rir_{rir_index:06d}"
            rng = random.Random(args.seed + scene_index * 100003 + local_index * 9176)
            prefer_obstructed = scene.scene_type_id in {1, 2, 3, 6} and (local_index % 2 == 1)
            placement = sample_placement(scene, rng, source_types=["rir_source"], prefer_obstructed=prefer_obstructed)
            case_dir = cases_dir / rir_id
            case_dir.mkdir(parents=True, exist_ok=True)
            row_seed = args.seed + scene_index * 100003 + local_index * 9176
            foa_path = scene_rir_dir / f"{rir_id}_foa.npy"
            mono_path = scene_rir_dir / f"{rir_id}_mono.npy"
            invalid_reason = ""
            try:
                raw_rir = backend.render_rir(
                    scene_mesh_path=geometry_dir / f"{scene.scene_id}.obj",
                    source_occ_xyz=placement.source_xyz,
                    receiver_occ_xyz=placement.receiver_xyz,
                    output_dir=case_dir,
                )
                foa, onset_sample = normalize_foa_rir(
                    raw_rir,
                    expected_samples,
                    align_output_onset=not args.preserve_propagation_delay,
                    onset_threshold_db=args.onset_threshold_db,
                )
                mono = np.ascontiguousarray((foa[0] * float(args.mono_scale)).astype(np.float32))
                np.save(foa_path, foa)
                np.save(mono_path, mono)
                valid, invalid_reason = validate_mono_rir(mono)
                metrics = compute_rir_metrics(mono, args.sample_rate) if args.compute_metrics else {}
            except Exception as exc:
                foa = np.zeros((4, expected_samples), dtype=np.float32)
                mono = np.zeros((expected_samples,), dtype=np.float32)
                onset_sample = 0
                np.save(foa_path, foa)
                np.save(mono_path, mono)
                valid = False
                invalid_reason = f"render_failed:{type(exc).__name__}:{exc}"
                metrics = compute_rir_metrics(mono, args.sample_rate) if args.compute_metrics else {}

            sx, sy, sz = placement.source_xyz
            rx, ry, rz = placement.receiver_xyz
            row: dict[str, Any] = {
                "rir_id": rir_id,
                "scene_id": scene.scene_id,
                "scenario_family": scene.scene_type,
                "split": "rir_bank",
                "random_seed": row_seed,
                "simulator_version": simulator_version(availability),
                "sample_rate": args.sample_rate,
                "ir_duration": args.ir_duration,
                "ray_count": args.indirect_ray_count,
                "direct_ray_count": args.direct_ray_count,
                "indirect_ray_count": args.indirect_ray_count,
                "align_output_onset": not args.preserve_propagation_delay,
                "onset_threshold_db": args.onset_threshold_db,
                "onset_sample": onset_sample,
                "mono_scale": args.mono_scale,
                "rir_path_foa": str(foa_path.relative_to(out)),
                "rir_path_mono": str(mono_path.relative_to(out)),
                "rir_format": RIR_FORMAT if args.rir_format == "both" else "foa",
                "foa_channel_order": EXPECTED_FOA_ORDER,
                "foa_normalization": FOA_NORMALIZATION,
                "foa_shape": json.dumps([4, expected_samples]),
                "mono_derivation": EXPECTED_MONO_DERIVATION,
                "metric_channel": METRIC_CHANNEL,
                "source_position_x": sx,
                "source_position_y": sy,
                "source_position_z": sz,
                "receiver_position_x": rx,
                "receiver_position_y": ry,
                "receiver_position_z": rz,
                "receiver_yaw": placement.azimuth_rad,
                "distance": placement.distance_m,
                "is_los": bool(placement.is_los),
                "is_occluded": not bool(placement.is_los),
                "occlusion_type": occlusion_type(placement),
                "occlusion_severity": int(placement.obstruction_count),
                "valid": bool(valid),
                "invalid_reason": invalid_reason,
            }
            for name in ["rt60", "edt", "early_late_ratio", "c50", "d50", "hf_loss", "peak_delay_ms", "rir_energy"]:
                row[name] = float(metrics.get(name, float("nan")))
            rows.append(row)
            if tqdm is not None and hasattr(progress, "update"):
                progress.update(1)
            progress_index += 1

    if tqdm is not None and hasattr(progress, "close"):
        progress.close()

    write_manifest_csv(out / "rir_manifest.csv", rows)
    write_manifest_jsonl(out / "rir_manifest.jsonl", rows)
    metric_summary = summarize_metric_rows(rows)
    (reports_dir / "metric_summary.json").write_text(json.dumps(metric_summary, indent=2, allow_nan=True), encoding="utf-8")
    report, invalid_rows, errors = validate_manifest_rows(rows, out)
    if rows and int(metric_summary["valid_count"]) == 0:
        errors.append("all_rirs_invalid")
        report["errors"] = errors
        report["passed"] = False
    report["output_dir"] = str(out)
    (reports_dir / "verification_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_invalid_csv(reports_dir / "invalid_rirs.csv", invalid_rows)

    summary = {
        "output_dir": str(out),
        "requested_rirs": sum(targets),
        "rendered_rows": len(rows),
        "valid_count": metric_summary["valid_count"],
        "invalid_count": metric_summary["invalid_count"],
        "manifest_csv": str(out / "rir_manifest.csv"),
        "verification_passed": not errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 2 if errors else 0


def write_invalid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
