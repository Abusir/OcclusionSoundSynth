from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import soundfile as sf

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
from soundspaces_adapter.validation import validate_rir_physics, write_validation_report
from soundspaces_adapter.visualize_debug import plot_geometry_debug, plot_rir_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end SoundSpaces backend verification.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/backend_verification"))
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--ray-count", type=int, default=1000)
    return parser.parse_args()


def find_placement(scene, rng: random.Random, want_los: bool):
    for _ in range(120):
        placement = sample_placement(scene, rng, source_types=["fire"], prefer_obstructed=not want_los)
        if placement.is_los == want_los:
            return placement
    return placement


def rir_channel_axis(rir: np.ndarray) -> tuple[int | None, int | None]:
    arr = np.asarray(rir)
    if arr.ndim != 2:
        return None, None
    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        return 0, int(arr.shape[0])
    return 1, int(arr.shape[1])


def wav_summary(path: Path) -> dict[str, object]:
    data, sample_rate = sf.read(path, always_2d=True)
    return {
        "path": str(path),
        "exists": path.exists(),
        "sample_rate": int(sample_rate),
        "shape": list(data.shape),
        "channels": int(data.shape[1]),
        "peak": float(np.max(np.abs(data))) if data.size else 0.0,
        "finite": bool(np.all(np.isfinite(data))),
    }


def render_case(
    backend: SoundSpacesBackend,
    scene,
    want_los: bool,
    args: argparse.Namespace,
    out: Path,
) -> dict[str, object]:
    case = "los" if want_los else "nlos"
    geometry_dir = out / "geometry"
    figure_dir = out / "figures"
    audio_dir = out / "audio"
    report_dir = out / "reports"
    rng = random.Random(args.seed + (11 if want_los else 29))
    placement = find_placement(scene, rng, want_los=want_los)
    assert_round_trip(placement.receiver_xyz)
    assert_round_trip(placement.source_xyz)
    files = export_scene_obj(scene, geometry_dir)
    plot_geometry_debug(
        scene,
        placement,
        figure_dir / f"{case}_{scene.scene_id}_geometry.png",
        title=f"{case.upper()} verification geometry",
    )
    rir = backend.render_rir(Path(files["obj"]), placement.source_xyz, placement.receiver_xyz, out)
    np.save(report_dir / f"{case}_{scene.scene_id}_rir.npy", rir)
    plot_rir_debug(
        rir,
        placement.source_xyz,
        placement.receiver_xyz,
        args.sample_rate,
        figure_dir / f"{case}_{scene.scene_id}_rir.png",
        title=f"{case.upper()} SoundSpaces RIR",
    )
    validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, placement.is_los)
    dry = synthesize_dry_sound("fire", args.sample_rate, args.duration, args.seed)
    audio_meta = backend.convolve_and_save(
        dry,
        rir,
        audio_dir / f"{case}_{scene.scene_id}_foa.wav",
        audio_dir / f"{case}_{scene.scene_id}_mono.wav",
    )
    foa_summary = wav_summary(Path(audio_meta["foa_wav_path"]))
    mono_summary = wav_summary(Path(audio_meta["mono_wav_path"]))
    channel_axis, channel_count = rir_channel_axis(rir)
    return {
        "case": case,
        "scene_id": scene.scene_id,
        "is_los": placement.is_los,
        "obstruction_types": placement.obstruction_types,
        "source_occ_xyz": placement.source_xyz,
        "receiver_occ_xyz": placement.receiver_xyz,
        "source_habitat_xyz": occ_to_habitat(placement.source_xyz).tolist(),
        "receiver_habitat_xyz": occ_to_habitat(placement.receiver_xyz).tolist(),
        "rir_shape": list(np.asarray(rir).shape),
        "rir_channel_axis": channel_axis,
        "rir_channel_count": channel_count,
        "rir_nonzero": bool(np.any(np.abs(rir) > 0.0)),
        "validation": validation.to_dict(),
        "audio": audio_meta,
        "foa_wav": foa_summary,
        "mono_wav": mono_summary,
        "passed": bool(
            validation.passed
            and channel_count == 4
            and np.any(np.abs(rir) > 0.0)
            and foa_summary["channels"] == 4
            and mono_summary["channels"] == 1
            and foa_summary["peak"] > 0.0
            and mono_summary["peak"] > 0.0
            and foa_summary["finite"]
            and mono_summary["finite"]
        ),
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    for path in (out / "geometry", out / "figures", out / "audio", out / "reports"):
        path.mkdir(parents=True, exist_ok=True)

    availability = check_soundspaces_available()
    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        indirect_ray_count=args.ray_count,
        output_directory=str(out),
    )
    config.save_json(out / "reports" / "soundspaces_config.json")
    report: dict[str, object] = {
        "soundspaces_available": availability,
        "config": config.to_dict(),
        "checks": {
            "import": "habitat_sim exposes audio APIs.",
            "los": "LOS RIR direct peak matches geometric delay.",
            "nlos": "NLOS RIR renders nonzero four-channel Ambisonics.",
            "audio": "FOA and mono WAV files are finite and nonempty.",
        },
        "material_note": "Programmatic OBJ scenes currently use SoundSpaces default material because no semantic mesh descriptor is generated yet.",
    }
    if not availability["available"]:
        report["status"] = "soundspaces_unavailable"
        report["passed"] = False
        write_validation_report(out / "reports" / "verification_report.json", report)
        print(json.dumps(report, indent=2))
        return 3

    try:
        scenes = generate_all_scenes(variants_per_type=10, seed=args.seed)
        los_scene = scenes[3]
        nlos_scene = scenes[0]
        backend = SoundSpacesBackend(config)
        cases = [
            render_case(backend, los_scene, True, args, out),
            render_case(backend, nlos_scene, False, args, out),
        ]
        report["status"] = "rendered"
        report["cases"] = cases
        report["passed"] = bool(all(case["passed"] for case in cases))
        write_validation_report(out / "reports" / "verification_report.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    except SoundSpacesUnavailableError as exc:
        report["status"] = "soundspaces_api_mismatch"
        report["message"] = str(exc)
        report["passed"] = False
        write_validation_report(out / "reports" / "verification_report.json", report)
        print(json.dumps(report, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
