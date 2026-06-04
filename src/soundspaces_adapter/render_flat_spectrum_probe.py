from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import soundfile as sf

SPECTROGRAM_DB_VMIN = -120.0
SPECTROGRAM_DB_VMAX = 20.0

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None
    FontProperties = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import sample_placement
from legacy_geometric.occ_synth.scene_generator import generate_all_scenes
from legacy_geometric.occ_synth.visualization import plot_scene
from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.material_database import (
    scene_material_assignment,
    write_occ_material_database,
    write_scene_material_assignments,
)
from soundspaces_adapter.validation import validate_rir_physics


FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"),
]
FONT_PATH = next((path for path in FONT_CANDIDATES if path.exists()), None)
FONT = FontProperties(fname=str(FONT_PATH)) if FontProperties is not None and FONT_PATH is not None else None

if plt is not None and FONT is not None:
    plt.rcParams.update(
        {
            "font.family": FONT.get_name(),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _font_kwargs() -> dict[str, Any]:
    return {"fontproperties": FONT} if FONT is not None else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 10-second flat-spectrum probe through generated OCC SoundSpaces scenes."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated_soundspaces_runs/flat_spectrum_probe_10s"))
    parser.add_argument("--variants-per-type", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--mel-bins", type=int, default=80)
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--save-logmel", action="store_true", help="Also save log-mel plots in addition to STFT plots.")
    parser.add_argument("--direct-ray-count", type=int, default=500)
    parser.add_argument("--indirect-ray-count", type=int, default=1000)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--onset-threshold-db", type=float, default=-80.0)
    parser.add_argument(
        "--disable-materials",
        action="store_true",
        help="Disable the default official RLR material database path.",
    )
    parser.add_argument(
        "--preserve-propagation-delay",
        action="store_true",
        help="Keep physical propagation delay instead of aligning output to the first RIR onset.",
    )
    parser.add_argument(
        "--prefer-obstructed-prob",
        type=float,
        default=0.65,
        help="For baffle/corridor/forest scenes, probability of preferring NLOS placement.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def flat_spectrum_probe(sample_rate: int, duration_s: float, seed: int) -> np.ndarray:
    """Create a deterministic real signal whose rFFT magnitudes are flat before normalization."""

    n = int(round(sample_rate * duration_s))
    if n <= 1:
        raise ValueError("duration must produce at least two samples")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, n // 2 + 1)
    spectrum = np.exp(1j * phases).astype(np.complex128)
    spectrum[0] = 1.0 + 0.0j
    if n % 2 == 0:
        spectrum[-1] = 1.0 + 0.0j
    signal = np.fft.irfft(spectrum, n=n).astype(np.float32)
    signal -= float(np.mean(signal))
    peak = float(np.max(np.abs(signal)))
    if peak > 0.0:
        signal *= 0.95 / peak
    return signal.astype(np.float32)


def spectrum_summary(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"rms": 0.0, "peak": 0.0, "spectral_centroid_hz": float("nan"), "hf_ratio_2k_nyquist": float("nan")}
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    power = spec * spec
    total = float(np.sum(power))
    centroid = float(np.sum(freqs * power) / total) if total > 0.0 else float("nan")
    hf = float(np.sum(power[freqs >= 2000.0]) / total) if total > 0.0 else float("nan")
    return {
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak": float(np.max(np.abs(x))),
        "spectral_centroid_hz": centroid,
        "hf_ratio_2k_nyquist": hf,
    }


def save_spectrum_plot(audio: np.ndarray, sample_rate: int, path: Path, title: str) -> None:
    if plt is None:
        return
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    ax.plot(freqs, db, linewidth=0.8)
    ax.set_xlim(0, sample_rate / 2)
    ax.set_xlabel("Frequency (Hz)", **_font_kwargs())
    ax.set_ylabel("Magnitude (dB)", **_font_kwargs())
    ax.set_title(title, **_font_kwargs())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for mel_index in range(n_mels):
        left, center, right = bin_points[mel_index : mel_index + 3]
        if center > left:
            filters[mel_index, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            filters[mel_index, center:right] = (right - np.arange(center, right)) / float(right - center)
    return filters


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))

    frame_count = 1 + max(0, (x.size - n_fft) // hop_length)
    shape = (frame_count, n_fft)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)) ** 2
    mel_power = mel_filterbank(sample_rate, n_fft, n_mels) @ power.T
    return (10.0 * np.log10(np.maximum(mel_power, 1e-12))).astype(np.float32)


def save_log_mel_plot(
    audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    if plt is None:
        return
    log_mel = log_mel_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    duration_s = len(np.asarray(audio).reshape(-1)) / float(sample_rate)
    image = ax.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration_s, 0, n_mels),
        cmap="magma",
        vmin=SPECTROGRAM_DB_VMIN,
        vmax=SPECTROGRAM_DB_VMAX,
    )
    ax.set_xlabel("Time (s)", **_font_kwargs())
    ax.set_ylabel("Mel bin", **_font_kwargs())
    ax.set_title(title, **_font_kwargs())
    fig.colorbar(image, ax=ax, label="Log mel power (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def log_stft_spectrogram(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))

    frame_count = 1 + max(0, (x.size - n_fft) // hop_length)
    shape = (frame_count, n_fft)
    strides = (x.strides[0] * hop_length, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = np.hanning(n_fft)
    magnitude = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1)).T
    return (20.0 * np.log10(np.maximum(magnitude, 1e-12))).astype(np.float32)


def save_stft_plot(
    audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    n_fft: int,
    hop_length: int,
) -> None:
    if plt is None:
        return
    log_stft = log_stft_spectrogram(audio, n_fft, hop_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    duration_s = len(np.asarray(audio).reshape(-1)) / float(sample_rate)
    image = ax.imshow(
        log_stft,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration_s, 0.0, sample_rate / 2.0),
        cmap="magma",
        vmin=SPECTROGRAM_DB_VMIN,
        vmax=SPECTROGRAM_DB_VMAX,
    )
    ax.set_xlabel("Time (s)", **_font_kwargs())
    ax.set_ylabel("Frequency (Hz)", **_font_kwargs())
    ax.set_title(title, **_font_kwargs())
    fig.colorbar(image, ax=ax, label="Log magnitude (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_delta_stft_plot(
    reference_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    n_fft: int,
    hop_length: int,
) -> None:
    if plt is None:
        return
    ref = log_stft_spectrogram(reference_audio, n_fft, hop_length)
    proc = log_stft_spectrogram(processed_audio, n_fft, hop_length)
    frame_count = min(ref.shape[1], proc.shape[1])
    delta = proc[:, :frame_count] - ref[:, :frame_count]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    duration_s = frame_count * hop_length / float(sample_rate)
    image = ax.imshow(
        delta,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration_s, 0.0, sample_rate / 2.0),
        cmap="coolwarm",
        vmin=SPECTROGRAM_DB_VMIN,
        vmax=SPECTROGRAM_DB_VMAX,
    )
    ax.set_xlabel("Time (s)", **_font_kwargs())
    ax.set_ylabel("Frequency (Hz)", **_font_kwargs())
    ax.set_title(title, **_font_kwargs())
    fig.colorbar(image, ax=ax, label="Output - input STFT (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_transfer_plot(
    reference_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
) -> None:
    if plt is None:
        return
    ref = np.asarray(reference_audio, dtype=np.float64).reshape(-1)
    proc = np.asarray(processed_audio, dtype=np.float64).reshape(-1)
    n = min(ref.size, proc.size)
    ref = ref[:n]
    proc = proc[:n]
    ref_mag = np.abs(np.fft.rfft(ref))
    proc_mag = np.abs(np.fft.rfft(proc))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    transfer_db = 20.0 * np.log10(np.maximum(proc_mag, 1e-12) / np.maximum(ref_mag, 1e-12))
    if transfer_db.size >= 101:
        kernel = np.ones(101, dtype=np.float64) / 101.0
        transfer_db = np.convolve(transfer_db, kernel, mode="same")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    ax.plot(freqs, transfer_db, linewidth=0.9)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xlim(0, sample_rate / 2)
    ax.set_ylim(-60, 20)
    ax.set_xlabel("Frequency (Hz)", **_font_kwargs())
    ax.set_ylabel("Output / input magnitude (dB)", **_font_kwargs())
    ax.set_title(title, **_font_kwargs())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if not 0.0 <= args.prefer_obstructed_prob <= 1.0:
        raise ValueError("--prefer-obstructed-prob must be in [0, 1]")

    availability = check_soundspaces_available()
    if not availability.get("available"):
        print(json.dumps({"passed": False, "reason": "soundspaces_unavailable", "detail": availability}, indent=2))
        return 2

    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    audio_dir = out / "audio"
    dry_dir = out / "dry"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    cases_dir = out / "cases"
    rirs_dir = out / "rirs"
    for path in (geometry_dir, audio_dir, dry_dir, figure_dir, report_dir, cases_dir, rirs_dir):
        path.mkdir(parents=True, exist_ok=True)

    material_db_path = report_dir / "occ_rlr_materials.json"
    material_db = write_occ_material_database(material_db_path)
    material_assignment_path = report_dir / "occ_scene_material_assignments.json"
    material_assignments = write_scene_material_assignments(material_assignment_path)
    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        direct_ray_count=args.direct_ray_count,
        indirect_ray_count=args.indirect_ray_count,
        thread_count=args.thread_count,
        output_directory=str(out),
        align_output_onset=not args.preserve_propagation_delay,
        onset_threshold_db=args.onset_threshold_db,
        enable_materials=not args.disable_materials,
        audio_materials_json=str(material_db_path) if not args.disable_materials else None,
        enable_rgb=False,
        enable_depth=False,
    )
    config.save_json(report_dir / "soundspaces_config.json")

    dry = flat_spectrum_probe(args.sample_rate, args.duration, args.seed + 7919)
    dry_wav = dry_dir / "flat_spectrum_probe.wav"
    sf.write(dry_wav, dry, args.sample_rate)
    np.save(dry_dir / "flat_spectrum_probe.npy", dry)
    save_spectrum_plot(dry, args.sample_rate, figure_dir / "flat_spectrum_probe_spectrum.png", "Flat-spectrum probe")
    save_stft_plot(
        dry,
        args.sample_rate,
        figure_dir / "flat_spectrum_probe_stft.png",
        "Flat-spectrum probe STFT",
        args.stft_n_fft,
        args.stft_hop_length,
    )
    if args.save_logmel:
        save_log_mel_plot(
            dry,
            args.sample_rate,
            figure_dir / "flat_spectrum_probe_logmel.png",
            "Flat-spectrum probe log-mel",
            args.stft_n_fft,
            args.stft_hop_length,
            args.mel_bins,
        )
    dry_metrics = spectrum_summary(dry, args.sample_rate)

    scenes = generate_all_scenes(variants_per_type=args.variants_per_type, seed=args.seed)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    backend = SoundSpacesBackend(config)
    placement_rng = random.Random(args.seed + 1009)

    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        files = export_scene_obj(scene, geometry_dir)
        prefer_obstructed = scene.scene_type_id in {1, 2, 3, 6} and placement_rng.random() < args.prefer_obstructed_prob
        placement = sample_placement(
            scene,
            placement_rng,
            source_types=["flat_spectrum_probe"],
            prefer_obstructed=prefer_obstructed,
        )
        case_id = f"probe_{scene_index:03d}_{scene.scene_id}"
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        rir = backend.render_rir(
            scene_mesh_path=Path(files["obj"]),
            source_occ_xyz=placement.source_xyz,
            receiver_occ_xyz=placement.receiver_xyz,
            output_dir=case_dir,
        )
        rir_path = rirs_dir / f"{case_id}_rir.npy"
        np.save(rir_path, np.asarray(rir, dtype=np.float32))
        validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, placement.is_los)

        foa_path = audio_dir / f"{case_id}_foa.wav"
        mono_path = audio_dir / f"{case_id}_mono.wav"
        audio_meta = backend.convolve_and_save(dry, rir, foa_path, mono_path)
        mono, sr = sf.read(mono_path, always_2d=False)
        if int(sr) != int(args.sample_rate):
            raise RuntimeError(f"unexpected sample rate in {mono_path}: {sr}")
        mono_metrics = spectrum_summary(np.asarray(mono, dtype=np.float32), args.sample_rate)
        if not args.no_plots:
            save_spectrum_plot(
                np.asarray(mono, dtype=np.float32),
                args.sample_rate,
                figure_dir / f"{case_id}_mono_spectrum.png",
                f"{scene.scene_type} mono output spectrum",
            )
            save_stft_plot(
                np.asarray(mono, dtype=np.float32),
                args.sample_rate,
                figure_dir / f"{case_id}_mono_stft.png",
                f"{scene.scene_type} mono output STFT",
                args.stft_n_fft,
                args.stft_hop_length,
            )
            save_delta_stft_plot(
                dry,
                np.asarray(mono, dtype=np.float32),
                args.sample_rate,
                figure_dir / f"{case_id}_delta_stft.png",
                f"{scene.scene_type} output-input STFT",
                args.stft_n_fft,
                args.stft_hop_length,
            )
            save_transfer_plot(
                dry,
                np.asarray(mono, dtype=np.float32),
                args.sample_rate,
                figure_dir / f"{case_id}_transfer.png",
                f"{scene.scene_type} transfer magnitude",
            )
            if args.save_logmel:
                save_log_mel_plot(
                    np.asarray(mono, dtype=np.float32),
                    args.sample_rate,
                    figure_dir / f"{case_id}_mono_logmel.png",
                    f"{scene.scene_type} mono output log-mel",
                    args.stft_n_fft,
                    args.stft_hop_length,
                    args.mel_bins,
                )
            plot_scene(scene, placement, figure_dir / f"{case_id}_layout.png")

        row = {
            "case_id": case_id,
            "scene_id": scene.scene_id,
            "scene_type": scene.scene_type,
            "variant_index": scene.variant_index,
            "is_outdoor": bool(scene.is_outdoor),
            "floor_material": scene_material_assignment(scene.scene_type)["floor"],
            "wall_material": scene_material_assignment(scene.scene_type)["wall"],
            "ceiling_material": scene_material_assignment(scene.scene_type)["ceiling"],
            "obstacle_material": scene_material_assignment(scene.scene_type)["obstacle"] or "",
            "open_boundary_material": scene_material_assignment(scene.scene_type)["open_boundary"] or "",
            "open_ceiling_material": scene_material_assignment(scene.scene_type)["open_ceiling"] or "",
            "material_source": "facebookresearch/rlr-audio-propagation mp3d_material_config.json",
            "prefer_obstructed": bool(prefer_obstructed),
            "is_los": bool(placement.is_los),
            "obstruction_count": int(placement.obstruction_count),
            "obstruction_types": "|".join(placement.obstruction_types),
            "distance_m": float(placement.distance_m),
            "azimuth_rad": float(placement.azimuth_rad),
            "elevation_rad": float(placement.elevation_rad),
            "source_x": float(placement.source_xyz[0]),
            "source_y": float(placement.source_xyz[1]),
            "source_z": float(placement.source_xyz[2]),
            "receiver_x": float(placement.receiver_xyz[0]),
            "receiver_y": float(placement.receiver_xyz[1]),
            "receiver_z": float(placement.receiver_xyz[2]),
            "obj_path": str(Path(files["obj"]).relative_to(out)),
            "rir_path": str(rir_path.relative_to(out)),
            "foa_wav_path": str(foa_path.relative_to(out)),
            "mono_wav_path": str(mono_path.relative_to(out)),
            "rir_shape": json.dumps(list(np.asarray(rir).shape)),
            "validation_passed": bool(validation.passed),
            "validation_expected_direct_delay_sample": int(validation.expected_direct_delay_sample),
            "validation_observed_first_peak_sample": validation.observed_first_peak_sample,
            "dry_rms": dry_metrics["rms"],
            "dry_peak": dry_metrics["peak"],
            "dry_spectral_centroid_hz": dry_metrics["spectral_centroid_hz"],
            "dry_hf_ratio_2k_nyquist": dry_metrics["hf_ratio_2k_nyquist"],
            "mono_rms": mono_metrics["rms"],
            "mono_peak": mono_metrics["peak"],
            "mono_spectral_centroid_hz": mono_metrics["spectral_centroid_hz"],
            "mono_hf_ratio_2k_nyquist": mono_metrics["hf_ratio_2k_nyquist"],
            "alignment_onset_sample": (audio_meta.get("alignment") or {}).get("onset_sample", ""),
        }
        rows.append(row)
        print(json.dumps({"rendered": len(rows), "case_id": case_id, "is_los": placement.is_los}, ensure_ascii=False))

    write_manifest(out / "probe_manifest.csv", rows)
    write_json(out / "probe_manifest.json", rows)
    write_json(
        report_dir / "run_summary.json",
        {
            "output_dir": str(out),
            "scene_count": len(scenes),
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "dry_wav": str(dry_wav),
            "material_database": str(material_db_path),
            "material_database_payload": material_db,
            "material_assignments": str(material_assignment_path),
            "material_assignments_payload": material_assignments,
            "soundspaces_config": config.to_dict(),
            "dry_metrics": dry_metrics,
            "soundspaces": availability,
            "manifest_csv": str(out / "probe_manifest.csv"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
