from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soundspaces_adapter.analyze_rir_impulse_probe import mono_rir_from_foa, normalize_rir_shape
from soundspaces_adapter.render_flat_spectrum_probe import (
    log_mel_spectrogram,
    save_log_mel_plot,
    save_spectrum_plot,
    save_transfer_plot,
    spectrum_summary,
    write_manifest,
)


DEFAULT_INPUT_DIR = Path(
    "generated_soundspaces_runs/"
    "six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir_rt60"
)
DEFAULT_FIRE_AUDIO = Path("git_version/examples/test_audio_bank/esc50_crackling_fire.wav")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a real fire sound through an existing six-scene SoundSpaces RIR manifest."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_soundspaces_runs/fire_sound_demo"))
    parser.add_argument("--fire-audio", type=Path, default=DEFAULT_FIRE_AUDIO)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--mel-bins", type=int, default=80)
    parser.add_argument("--listening-peak", type=float, default=0.95)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    source_t = np.arange(audio.shape[0], dtype=np.float64) / float(source_rate)
    target_n = int(round(audio.shape[0] * target_rate / source_rate))
    target_t = np.arange(target_n, dtype=np.float64) / float(target_rate)
    return np.interp(target_t, source_t, audio).astype(np.float32)


def mono_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.mean(arr, axis=1, dtype=np.float32)
    raise ValueError(f"expected 1D or 2D audio, got shape {arr.shape}")


def extend_to_duration(audio: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    target_n = int(round(sample_rate * duration_s))
    if target_n <= 0:
        raise ValueError("--duration must be positive")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("fire audio is empty")
    x = x.astype(np.float32, copy=True)
    x -= float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak > 0.0:
        x *= 0.95 / peak
    if x.size >= target_n:
        return x[:target_n].astype(np.float32)
    padded = np.zeros(target_n, dtype=np.float32)
    padded[: x.size] = x
    return padded


def convolve_mono(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    y = np.convolve(np.asarray(source, dtype=np.float64), np.asarray(rir, dtype=np.float64), mode="full")
    return y[: source.shape[0]].astype(np.float32)


def convolve_foa(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    arr = normalize_rir_shape(rir)
    channels = [convolve_mono(source, arr[:, channel]) for channel in range(arr.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def normalized_for_listening(audio: np.ndarray, peak_target: float) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0.0:
        return arr.copy()
    return (arr * (peak_target / peak)).astype(np.float32)


def save_delta_log_mel_plot(
    reference_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    if plt is None:
        return
    ref = log_mel_spectrogram(reference_audio, sample_rate, n_fft, hop_length, n_mels)
    proc = log_mel_spectrogram(processed_audio, sample_rate, n_fft, hop_length, n_mels)
    frame_count = min(ref.shape[1], proc.shape[1])
    delta = proc[:, :frame_count] - ref[:, :frame_count]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    duration_s = frame_count * hop_length / float(sample_rate)
    image = ax.imshow(
        delta,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration_s, 0, n_mels),
        cmap="coolwarm",
        vmin=-30.0,
        vmax=30.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Receiver - source log-Mel (dB)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if plt is None or not rows:
        return
    labels = [str(row["scene_type"]) for row in rows]
    source_rms = float(rows[0]["source_rms"])
    receiver_rms = np.array([float(row["receiver_rms"]) for row in rows], dtype=np.float64)
    attenuation_db = 20.0 * np.log10(np.maximum(receiver_rms, 1e-12) / max(source_rms, 1e-12))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=140)
    x = np.arange(len(labels))
    ax.bar(x, attenuation_db, color="#3f6fb5")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Receiver / source RMS (dB)")
    ax.set_title("Fire sound level after scene RIR")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = input_dir / "probe_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not args.fire_audio.exists():
        raise FileNotFoundError(args.fire_audio)

    fire_raw, fire_rate = sf.read(args.fire_audio, always_2d=False)
    fire = mono_audio(fire_raw)
    fire = resample_linear(fire, int(fire_rate), args.sample_rate)
    source_10s = extend_to_duration(fire, args.sample_rate, args.duration)

    dry_dir = output_dir / "dry"
    audio_dir = output_dir / "audio"
    figure_dir = output_dir / "figures"
    dry_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dry_path = dry_dir / "fire_source_10s.wav"
    sf.write(dry_path, source_10s, args.sample_rate)
    if not args.no_plots:
        save_spectrum_plot(source_10s, args.sample_rate, figure_dir / "fire_source_10s_spectrum.png", "Fire source spectrum")
        save_log_mel_plot(
            source_10s,
            args.sample_rate,
            figure_dir / "fire_source_10s_logmel.png",
            "Fire source log-Mel",
            args.stft_n_fft,
            args.stft_hop_length,
            args.mel_bins,
        )

    source_stats = spectrum_summary(source_10s, args.sample_rate)
    rows_out: list[dict[str, Any]] = []
    for row in read_manifest(manifest_path):
        case_id = row["case_id"]
        rir_path = input_dir / row["rir_path"]
        rir = np.load(rir_path)
        mono_rir = mono_rir_from_foa(rir)
        receiver = convolve_mono(source_10s, mono_rir)
        foa = convolve_foa(source_10s, rir)

        receiver_path = audio_dir / f"{case_id}_fire_mono.wav"
        receiver_listen_path = audio_dir / f"{case_id}_fire_mono_listening_norm.wav"
        foa_path = audio_dir / f"{case_id}_fire_foa.wav"
        sf.write(receiver_path, receiver, args.sample_rate)
        sf.write(receiver_listen_path, normalized_for_listening(receiver, args.listening_peak), args.sample_rate)
        sf.write(foa_path, foa, args.sample_rate)

        if not args.no_plots:
            save_spectrum_plot(
                receiver,
                args.sample_rate,
                figure_dir / f"{case_id}_fire_receiver_spectrum.png",
                f"{row['scene_type']} fire receiver spectrum",
            )
            save_log_mel_plot(
                receiver,
                args.sample_rate,
                figure_dir / f"{case_id}_fire_receiver_logmel.png",
                f"{row['scene_type']} fire receiver log-Mel",
                args.stft_n_fft,
                args.stft_hop_length,
                args.mel_bins,
            )
            save_transfer_plot(
                source_10s,
                receiver,
                args.sample_rate,
                figure_dir / f"{case_id}_fire_receiver_source_transfer.png",
                f"{row['scene_type']} fire receiver/source transfer",
            )
            save_delta_log_mel_plot(
                source_10s,
                receiver,
                args.sample_rate,
                figure_dir / f"{case_id}_fire_receiver_minus_source_logmel.png",
                f"{row['scene_type']} fire receiver - source log-Mel",
                args.stft_n_fft,
                args.stft_hop_length,
                args.mel_bins,
            )

        receiver_stats = spectrum_summary(receiver, args.sample_rate)
        output_row: dict[str, Any] = {
            "case_id": case_id,
            "scene_id": row.get("scene_id", ""),
            "scene_type": row.get("scene_type", ""),
            "target": row.get("target", ""),
            "actual": row.get("actual", ""),
            "is_los": row.get("is_los", ""),
            "distance_m": row.get("distance_m", ""),
            "rir_path": row.get("rir_path", ""),
            "fire_source_input_path": str(args.fire_audio),
            "fire_source_10s_path": str(dry_path.relative_to(output_dir)),
            "fire_receiver_mono_path": str(receiver_path.relative_to(output_dir)),
            "fire_receiver_mono_listening_norm_path": str(receiver_listen_path.relative_to(output_dir)),
            "fire_receiver_foa_path": str(foa_path.relative_to(output_dir)),
        }
        for key, value in source_stats.items():
            output_row[f"source_{key}"] = value
        for key, value in receiver_stats.items():
            output_row[f"receiver_{key}"] = value
        output_row["receiver_to_source_rms_db"] = float(
            20.0 * np.log10(max(receiver_stats["rms"], 1e-12) / max(source_stats["rms"], 1e-12))
        )
        rows_out.append(output_row)

    write_manifest(output_dir / "fire_sound_manifest.csv", rows_out)
    (output_dir / "fire_sound_manifest.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    if not args.no_plots:
        save_summary_plot(rows_out, figure_dir / "fire_receiver_level_summary.png")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "input_dir": str(input_dir),
                "fire_audio": str(args.fire_audio),
                "case_count": len(rows_out),
                "duration_s": args.duration,
                "sample_rate": args.sample_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
