from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import soundfile as sf

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_geometric.occ_synth.audio_library import AudioLibrary, load_audio_library, load_dry_audio
from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.pipeline import make_label, scene_geometry_record, write_json
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes
from legacy_geometric.occ_synth.source_config import SourceConfig
from legacy_geometric.occ_synth.validation import summarize_validation, validate_catalog, validate_label, validate_obj
from legacy_geometric.occ_synth.visualization import plot_catalog_overview, plot_scene, plot_waveform

from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.material_database import write_occ_material_database, write_scene_material_assignments
from soundspaces_adapter.materials import write_material_map
from soundspaces_adapter.validation import validate_rir_physics


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a labeled SoundSpaces dataset from OCC scenes and dry audio manifest.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_soundspaces_runs/dataset"))
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--source-dataset-name", type=str, default="fire_sound_dataset_v2")
    parser.add_argument("--variants-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--scene-sampling", choices=["sequential", "random"], default="random")
    parser.add_argument(
        "--audio-sampling",
        choices=["cover_once_then_random", "random_with_replacement", "sequential_without_replacement"],
        default="cover_once_then_random",
    )
    parser.add_argument("--audio-start-index", type=int, default=0)
    parser.add_argument("--example-start-index", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--duration",
        type=str,
        default="source",
        help="Output duration in seconds, or 'source' to use each source file's full duration.",
    )
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--ray-count", type=int, default=1000)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=160)
    parser.add_argument("--preserve-propagation-delay", action="store_true", help="Keep physical time-of-flight silence at the output start.")
    parser.add_argument("--onset-threshold-db", type=float, default=-80.0)
    parser.add_argument(
        "--disable-materials",
        action="store_true",
        help="Disable the default official RLR material database path.",
    )
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip examples whose label, FOA wav, and mono wav already exist.")
    return parser.parse_args(argv)


def pick_scene(rng: random.Random, scenes: list[Any], idx: int, mode: str):
    if mode == "random":
        return rng.choice(scenes)
    return scenes[idx % len(scenes)]


def find_placement(scene: Any, rng: random.Random, source_label: str, max_attempts: int):
    placement = None
    prefer_obstructed = scene.scene_type_id in {1, 2, 3, 6} and rng.random() < 0.65
    for _ in range(max_attempts):
        placement = sample_placement(scene, rng, source_types=[source_label], prefer_obstructed=prefer_obstructed)
        if placement is not None:
            return placement
    return placement


def pick_audio_item(audio_library: AudioLibrary, rng: random.Random, idx: int, mode: str, start_index: int):
    if mode == "cover_once_then_random":
        effective_index = start_index + idx
        if effective_index < len(audio_library):
            return audio_library.get(effective_index)
        return audio_library.sample(rng)
    if mode == "random_with_replacement":
        return audio_library.sample(rng)
    effective_index = start_index + idx
    if effective_index >= len(audio_library):
        raise IndexError(
            f"audio-start-index + num-examples exceeds manifest length: "
            f"{start_index} + {idx + 1} > {len(audio_library)}"
        )
    return audio_library.get(effective_index)


def parse_duration_spec(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"source", "full", "original", "auto"}:
        return None
    duration = float(normalized)
    if duration <= 0:
        raise ValueError("--duration must be positive seconds or 'source'")
    return duration


def expected_source_samples(item: Any, sample_rate: int, duration_s: float | None) -> int:
    if duration_s is not None:
        return int(round(sample_rate * duration_s))
    info = sf.info(item.path)
    if int(info.frames) <= 0:
        raise ValueError(f"dry audio is empty: {item.path}")
    if int(info.samplerate) == sample_rate:
        return int(info.frames)
    # scipy.signal.resample_poly returns ceil(n * target_sr / source_sr)
    # for the audio lengths used here. This mirrors load_dry_audio(None).
    return int(math.ceil(int(info.frames) * sample_rate / int(info.samplerate)))


def build_distillation_row(label: dict[str, Any]) -> dict[str, Any]:
    audio_input = label.get("audio_input") or {}
    audio_meta = audio_input.get("metadata") or {}
    files = label.get("files") or {}
    occlusion = label.get("occlusion") or {}
    relative = label.get("relative") or {}
    source = label.get("source") or {}
    receiver = label.get("receiver") or {}
    render = label.get("render") or {}
    return {
        "example_id": str(label.get("example_id", "")),
        "student_audio_path": str(files.get("audio_wav", "")),
        "student_mono_path": str(files.get("mono_wav", "")),
        "teacher_audio_path": str(audio_input.get("path", "")),
        "source_audio_id": str(audio_input.get("audio_id", "")),
        "source_dataset": str(audio_input.get("dataset", source.get("dataset", ""))),
        "source_label": str(audio_input.get("label", source.get("class", ""))),
        "source_clip_start_sec": audio_meta.get("start_sec", audio_meta.get("clip_start_sec", "")),
        "source_clip_duration_sec": audio_meta.get("duration_sec", audio_meta.get("clip_duration_sec", "")),
        "source_sample_rate": audio_meta.get("sample_rate", ""),
        "scene_id": str(label.get("scene_id", "")),
        "scene_index": label.get("scene_index", ""),
        "scene_type": str(label.get("scene_type", "")),
        "is_direct_path_obstructed": bool(occlusion.get("is_direct_path_obstructed", False)),
        "obstruction_count": occlusion.get("obstruction_count", 0),
        "distance_m": relative.get("distance_m", ""),
        "azimuth_deg": relative.get("azimuth_deg", ""),
        "receiver_x_m": (receiver.get("center_xyz_m") or ["", "", ""])[0],
        "receiver_y_m": (receiver.get("center_xyz_m") or ["", "", ""])[1],
        "receiver_z_m": (receiver.get("center_xyz_m") or ["", "", ""])[2],
        "source_x_m": (source.get("xyz_m") or ["", "", ""])[0],
        "source_y_m": (source.get("xyz_m") or ["", "", ""])[1],
        "source_z_m": (source.get("xyz_m") or ["", "", ""])[2],
        "teacher_role": "original_audio",
        "student_role": "simulated_audio",
        "label_json": str(files.get("label_json", "")),
        "render_backend": str(render.get("backend", "")),
    }


def write_distillation_index(output_dir: Path, labels: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [build_distillation_row(label) for label in labels]
    csv_path = output_dir / "distillation_index.csv"
    json_path = output_dir / "distillation_index.json"
    fieldnames = list(rows[0].keys()) if rows else [
        "example_id",
        "student_audio_path",
        "student_mono_path",
        "teacher_audio_path",
        "source_audio_id",
        "source_dataset",
        "source_label",
        "source_clip_start_sec",
        "source_clip_duration_sec",
        "source_sample_rate",
        "scene_id",
        "scene_index",
        "scene_type",
        "is_direct_path_obstructed",
        "obstruction_count",
        "distance_m",
        "azimuth_deg",
        "receiver_x_m",
        "receiver_y_m",
        "receiver_z_m",
        "source_x_m",
        "source_y_m",
        "source_z_m",
        "teacher_role",
        "student_role",
        "label_json",
        "render_backend",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)
    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "row_count": len(rows),
        "unique_source_audio_count": len({row["source_audio_id"] for row in rows if row["source_audio_id"]}),
        "occluded_count": sum(1 for row in rows if row["is_direct_path_obstructed"]),
        "clear_count": sum(1 for row in rows if not row["is_direct_path_obstructed"]),
    }


def _wav_matches(path: Path, expected_sample_rate: int, expected_samples: int) -> bool:
    if not path.exists():
        return False
    try:
        info = sf.info(path)
    except Exception:
        return False
    return int(info.samplerate) == expected_sample_rate and int(info.frames) == expected_samples


def load_completed_label(
    label_path: Path,
    expected_sample_rate: int,
    expected_samples: int,
    expected_alignment_enabled: bool,
) -> dict[str, Any] | None:
    if not label_path.exists():
        return None
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            label = json.load(handle)
    except Exception:
        return None
    files = label.get("files") or {}
    required_paths = [
        files.get("audio_wav"),
        files.get("mono_wav"),
    ]
    if not all(path and Path(path).exists() for path in required_paths):
        return None
    render = label.get("render") or {}
    if int(render.get("sample_rate", -1)) != expected_sample_rate:
        return None
    if int(render.get("num_samples", -1)) != expected_samples:
        return None
    if not _wav_matches(Path(required_paths[0]), expected_sample_rate, expected_samples):
        return None
    if not _wav_matches(Path(required_paths[1]), expected_sample_rate, expected_samples):
        return None
    alignment = ((label.get("render") or {}).get("audio_backend_meta") or {}).get("alignment") or {}
    if bool(alignment.get("enabled", False)) != bool(expected_alignment_enabled):
        return None
    return label


def cleanup_incomplete_example_outputs(
    example_id: str,
    label_path: Path,
    audio_dir: Path,
    figure_dir: Path,
    case_dir: Path,
) -> bool:
    paths = [
        label_path,
        audio_dir / f"{example_id}_foa.wav",
        audio_dir / f"{example_id}_mono.wav",
        figure_dir / f"{example_id}_layout.png",
        figure_dir / f"{example_id}_waveform.png",
    ]
    had_partial = any(path.exists() for path in paths) or case_dir.exists()
    if not had_partial:
        return False
    for path in paths:
        if path.exists():
            path.unlink()
    if case_dir.exists():
        shutil.rmtree(case_dir)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    availability = check_soundspaces_available()
    if not availability.get("available"):
        print(json.dumps({"passed": False, "reason": "soundspaces_unavailable", "detail": availability}, ensure_ascii=False, indent=2))
        return 2

    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    audio_dir = out / "audio"
    label_dir = out / "labels"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    for path in (geometry_dir, audio_dir, label_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    scenes = generate_all_scenes(variants_per_type=args.variants_per_type, seed=args.seed)
    validation_errors: list[str] = []
    validation_errors.extend(validate_catalog(scenes, args.variants_per_type))
    if not args.no_overview:
        plot_catalog_overview(scenes, figure_dir / "catalog_overview.png")

    audio_library: AudioLibrary = load_audio_library(args.audio_manifest.resolve(), dataset_name=args.source_dataset_name)
    duration_s = parse_duration_spec(args.duration)
    if args.audio_sampling in {"cover_once_then_random", "sequential_without_replacement"}:
        if args.audio_start_index < 0:
            raise ValueError("--audio-start-index must be >= 0")
    if args.example_start_index < 0:
        raise ValueError("--example-start-index must be >= 0")
    if args.audio_sampling == "sequential_without_replacement" and args.audio_start_index + args.num_examples > len(audio_library):
            raise ValueError(
                f"Requested sequential audio coverage exceeds manifest length: "
                f"start={args.audio_start_index}, num_examples={args.num_examples}, total={len(audio_library)}"
            )
    source_config = SourceConfig(
        classes=audio_library.classes,
        dataset_name=audio_library.dataset_name,
        manifest_path=str(args.audio_manifest.resolve()),
    )
    material_db_path = report_dir / "occ_rlr_materials.json"
    write_occ_material_database(material_db_path)
    material_assignment_path = report_dir / "occ_scene_material_assignments.json"
    write_scene_material_assignments(material_assignment_path)
    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        indirect_ray_count=args.ray_count,
        thread_count=args.thread_count,
        output_directory=str(out),
        align_output_onset=not args.preserve_propagation_delay,
        onset_threshold_db=args.onset_threshold_db,
        enable_materials=not args.disable_materials,
        audio_materials_json=str(material_db_path) if not args.disable_materials else None,
        enable_rgb=False,
        enable_depth=False,
    )
    config.save_json(report_dir / "soundspaces_dataset_config.json")
    backend = SoundSpacesBackend(config)
    rng = random.Random(args.seed + 1009)

    geometry_records = []
    labels: list[dict[str, Any]] = []
    scene_lookup = {scene.scene_id: scene for scene in scenes}
    for scene in scenes:
        geometry_records.append(scene_geometry_record(scene))
        files = export_scene_obj(scene, geometry_dir)
        write_material_map(geometry_dir / f"{scene.scene_id}.sound_materials.json")
        validation_errors.extend(validate_obj(Path(files["obj"])))

    example_reports: list[dict[str, Any]] = []
    incomplete_rerun_count = 0
    progress = (
        tqdm(
            range(args.num_examples),
            desc="SoundSpaces synth",
            unit="example",
            dynamic_ncols=True,
        )
        if tqdm is not None and not args.no_progress_bar
        else range(args.num_examples)
    )
    for idx in progress:
        example_index = args.example_start_index + idx
        scene = pick_scene(rng, scenes, idx, args.scene_sampling)
        audio_item = pick_audio_item(audio_library, rng, idx, args.audio_sampling, args.audio_start_index)
        expected_samples = expected_source_samples(audio_item, args.sample_rate, duration_s)
        placement = find_placement(scene, rng, audio_item.label, args.max_attempts)
        if placement is None:
            validation_errors.append(f"{scene.scene_id}: could not sample placement")
            if tqdm is not None and hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str("placement_failed")
            continue
        example_id = f"ss_example_{example_index:06d}_{scene.scene_id}"
        label_path = label_dir / f"{example_id}.json"
        case_dir = out / "cases" / example_id
        if args.resume:
            completed_label = load_completed_label(
                label_path,
                args.sample_rate,
                expected_samples,
                expected_alignment_enabled=not args.preserve_propagation_delay,
            )
            if completed_label is not None:
                validation_errors.extend(
                    validate_label(completed_label, scene_lookup[completed_label["scene_id"]], allowed_source_classes=set(source_config.classes))
                )
                labels.append(completed_label)
                example_reports.append(
                    {
                        "example_id": completed_label.get("example_id", example_id),
                        "scene_id": completed_label.get("scene_id", scene.scene_id),
                        "audio_id": (completed_label.get("audio_input") or {}).get("audio_id", audio_item.audio_id),
                        "source_label": (completed_label.get("audio_input") or {}).get("label", audio_item.label),
                        "is_los": bool((completed_label.get("relative") or {}).get("is_los", placement.is_los)),
                        "obstruction_count": (completed_label.get("occlusion") or {}).get(
                            "obstruction_count", placement.obstruction_count
                        ),
                        "resumed": True,
                    }
                )
                if tqdm is not None and hasattr(progress, "set_postfix"):
                    progress.set_postfix(rendered=len(labels), resumed=1, refresh=False)
                continue
            if cleanup_incomplete_example_outputs(example_id, label_path, audio_dir, figure_dir, case_dir):
                incomplete_rerun_count += 1

        dry_audio = load_dry_audio(audio_item, args.sample_rate, duration_s)
        if dry_audio.shape[0] != expected_samples:
            expected_samples = int(dry_audio.shape[0])
        scene_files = {
            "obj": str(geometry_dir / f"{scene.scene_id}.obj"),
            "mtl": str(geometry_dir / f"{scene.scene_id}.mtl"),
        }
        case_dir.mkdir(parents=True, exist_ok=True)
        rir = backend.render_rir(
            scene_mesh_path=Path(scene_files["obj"]),
            source_occ_xyz=placement.source_xyz,
            receiver_occ_xyz=placement.receiver_xyz,
            output_dir=case_dir,
        )
        validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, placement.is_los)
        if not validation.passed:
            validation_errors.append(f"{example_id}: invalid rir physics")
        foa_wav_path = audio_dir / f"{example_id}_foa.wav"
        mono_wav_path = audio_dir / f"{example_id}_mono.wav"
        audio_meta = backend.convolve_and_save(dry_audio, rir, foa_wav_path, mono_wav_path)
        render_meta = {
            "backend": "soundspaces2",
            "wav_path": str(foa_wav_path),
            "mono_wav_path": str(mono_wav_path),
            "sample_rate": args.sample_rate,
            "num_channels": int(audio_meta.get("channels", rir.shape[1] if rir.ndim == 2 else 1)),
            "mono_num_channels": 1,
            "num_samples": int(dry_audio.shape[0]),
            "rir_shape": list(rir.shape),
            "rir_nonzero_samples": int((abs(rir).sum(axis=1) > 0).sum()) if rir.ndim == 2 else int((abs(rir) > 0).sum()),
            "dry_source": "manifest_audio",
            "render_config": config.to_dict(),
            "validation": validation.to_dict(),
            "audio_backend_meta": audio_meta,
        }
        if "output_format" in audio_meta:
            render_meta["render_config"]["output_format"] = audio_meta["output_format"]
        if "channel_order" in audio_meta:
            render_meta["render_config"]["channel_order"] = audio_meta["channel_order"]
        if "mono_derivation" in audio_meta:
            render_meta["mono_derivation"] = audio_meta["mono_derivation"]
        files = dict(scene_files)
        files["audio_wav"] = str(foa_wav_path)
        files["mono_wav"] = str(mono_wav_path)
        layout_png = figure_dir / f"{example_id}_layout.png"
        plot_scene(scene, placement, layout_png)
        files["layout_png"] = str(layout_png)
        if idx < 6:
            wave_png = figure_dir / f"{example_id}_waveform.png"
            plot_waveform(foa_wav_path, wave_png)
            files["waveform_png"] = str(wave_png)

        label = make_label(example_id, scene, placement, files, render_meta, source_config, audio_item=audio_item)
        label["files"]["label_json"] = str(label_path)
        write_json(label_path, label)
        validation_errors.extend(validate_label(label, scene_lookup[label["scene_id"]], allowed_source_classes=set(source_config.classes)))
        labels.append(label)
        example_reports.append(
            {
                "example_id": example_id,
                "scene_id": scene.scene_id,
                "audio_id": audio_item.audio_id,
                "source_label": audio_item.label,
                "is_los": placement.is_los,
                "obstruction_count": placement.obstruction_count,
            }
        )
        if tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                rendered=len(labels),
                los=int(placement.is_los),
                label=audio_item.label,
                refresh=False,
            )

    if tqdm is not None and hasattr(progress, "close"):
        progress.close()

    write_json(out / "scene_geometry_catalog.json", geometry_records)
    write_json(out / "labels_index.json", labels)
    distillation_index = write_distillation_index(out, labels)
    remaining_unique = max(0, len(audio_library) - args.audio_start_index)
    if args.audio_sampling == "cover_once_then_random":
        covered_once_count = min(remaining_unique, max(0, len(labels)))
        random_tail_count = max(0, len(labels) - covered_once_count)
        guaranteed_unique_prefix_end_index = (
            args.audio_start_index + covered_once_count - 1 if covered_once_count > 0 else None
        )
    elif args.audio_sampling == "sequential_without_replacement":
        covered_once_count = len(labels)
        random_tail_count = 0
        guaranteed_unique_prefix_end_index = (
            args.audio_start_index + len(labels) - 1 if len(labels) > 0 else None
        )
    else:
        covered_once_count = 0
        random_tail_count = len(labels)
        guaranteed_unique_prefix_end_index = None

    summary = {
        "output_dir": str(out),
        "num_examples_requested": args.num_examples,
        "num_examples_rendered": len(labels),
        "num_examples_resumed": sum(1 for report in example_reports if report.get("resumed")),
        "num_incomplete_examples_rerun": incomplete_rerun_count,
        "scene_sampling": args.scene_sampling,
        "audio_sampling": args.audio_sampling,
        "audio_start_index": args.audio_start_index,
        "example_start_index": args.example_start_index,
        "duration": args.duration,
        "seed": args.seed,
        "audio_manifest": str(args.audio_manifest.resolve()),
        "audio_classes": list(audio_library.classes),
        "audio_manifest_length": len(audio_library),
        "audio_coverage_in_run": {
            "covered_once_count": covered_once_count,
            "random_tail_count": random_tail_count,
            "guaranteed_unique_prefix_end_index": guaranteed_unique_prefix_end_index,
        },
        "distillation_index": distillation_index,
        "example_reports": example_reports,
        "validation": summarize_validation(validation_errors),
    }
    write_json(out / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
