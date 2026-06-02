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

from soundspaces_adapter.render_flat_spectrum_probe import save_delta_stft_plot, save_stft_plot
from soundspaces_adapter.validation import SPEED_OF_SOUND, energy_envelope, expected_delay_sample, first_peak_sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved SoundSpaces RIRs with a single impulse probe.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("generated_soundspaces_runs/flat_spectrum_probe_10s"),
        help="Directory containing probe_manifest.csv and saved RIR .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_soundspaces_runs/rir_impulse_analysis"),
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--impulse-time", type=float, default=0.5, help="Impulse time in seconds; default is inside the first second.")
    parser.add_argument("--direct-window-ms", type=float, default=2.0)
    parser.add_argument("--early-window-ms", type=float, default=50.0)
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def normalize_rir_shape(rir: np.ndarray) -> np.ndarray:
    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D RIR, got shape {arr.shape}")
    return arr


def mono_rir_from_foa(rir: np.ndarray) -> np.ndarray:
    arr = normalize_rir_shape(rir)
    return arr[:, 0] * np.sqrt(2.0)


def make_impulse_probe(sample_rate: int, duration_s: float, impulse_time_s: float) -> np.ndarray:
    n = int(round(sample_rate * duration_s))
    if n <= 0:
        raise ValueError("duration must be positive")
    impulse_sample = int(round(sample_rate * impulse_time_s))
    if not 0 <= impulse_sample < n:
        raise ValueError("--impulse-time must fall inside the generated signal duration")
    audio = np.zeros(n, dtype=np.float32)
    audio[impulse_sample] = 0.95
    return audio


def convolve_impulse_probe(probe: np.ndarray, mono_rir: np.ndarray) -> np.ndarray:
    rendered = np.convolve(np.asarray(probe, dtype=np.float64), np.asarray(mono_rir, dtype=np.float64), mode="full")
    return rendered[: probe.shape[0]].astype(np.float32)


def db(value: float, floor: float = 1e-20) -> float:
    return float(10.0 * np.log10(max(value, floor)))


def estimate_decay_time_ms(edc_db: np.ndarray, sample_rate: int, low_db: float, high_db: float) -> float:
    # Fit between high_db and low_db, e.g. -5..-25 dB. Returns NaN when the RIR is too short or not decayed enough.
    indices = np.flatnonzero((edc_db <= high_db) & (edc_db >= low_db))
    if indices.size < 2:
        return float("nan")
    t = indices.astype(np.float64) / float(sample_rate)
    y = edc_db[indices]
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0.0:
        return float("nan")
    rt60 = -60.0 / float(slope)
    return float(rt60 * 1000.0)


def band_energy_ratio_db(mono_rir: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(1, mono_rir.size)))))
    response = np.fft.rfft(mono_rir, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(20.0 * np.log10(np.maximum(np.abs(response[mask]), 1e-12))))


def rir_metrics(rir: np.ndarray, row: dict[str, str], sample_rate: int, direct_window_ms: float, early_window_ms: float) -> dict[str, Any]:
    arr = normalize_rir_shape(rir)
    mono = mono_rir_from_foa(arr)
    envelope = energy_envelope(arr)
    energy = envelope * envelope
    total = float(np.sum(energy))
    peak = first_peak_sample(envelope)
    peak_value = float(np.max(envelope)) if envelope.size else 0.0
    distance = float(row["distance_m"])
    expected = expected_delay_sample(distance, sample_rate, SPEED_OF_SOUND)

    half_direct = int(round(sample_rate * direct_window_ms / 1000.0))
    direct_start = max(0, expected - half_direct)
    direct_stop = min(envelope.size, expected + half_direct + 1)
    early_stop = min(envelope.size, int(round(sample_rate * early_window_ms / 1000.0)))
    direct_energy = float(np.sum(energy[direct_start:direct_stop]))
    early_energy = float(np.sum(energy[:early_stop]))
    late_energy = float(np.sum(energy[early_stop:]))
    residual_energy = max(total - direct_energy, 0.0)

    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc, 1e-20) / max(float(edc[0]) if edc.size else 0.0, 1e-20))
    rt60_t10_ms = estimate_decay_time_ms(edc_db, sample_rate, -15.0, -5.0)
    rt60_t20_ms = estimate_decay_time_ms(edc_db, sample_rate, -25.0, -5.0)

    return {
        "case_id": row["case_id"],
        "scene_id": row["scene_id"],
        "scene_type": row["scene_type"],
        "variant_index": int(row["variant_index"]),
        "is_outdoor": row["is_outdoor"] == "True",
        "is_los": row["is_los"] == "True",
        "obstruction_count": int(row["obstruction_count"]),
        "obstruction_types": row["obstruction_types"],
        "distance_m": distance,
        "expected_delay_sample": expected,
        "expected_delay_ms": expected / sample_rate * 1000.0,
        "observed_first_peak_sample": peak if peak is not None else "",
        "observed_first_peak_ms": "" if peak is None else peak / sample_rate * 1000.0,
        "first_peak_delay_error_ms": "" if peak is None else (peak - expected) / sample_rate * 1000.0,
        "rir_channels": int(arr.shape[1]),
        "rir_samples": int(arr.shape[0]),
        "rir_duration_ms": arr.shape[0] / sample_rate * 1000.0,
        "peak_envelope": peak_value,
        "total_energy": total,
        "direct_energy": direct_energy,
        "early_energy_50ms": early_energy,
        "late_energy_after_50ms": late_energy,
        "direct_to_residual_db": db(direct_energy / max(residual_energy, 1e-20)),
        "c50_db": db(early_energy / max(late_energy, 1e-20)),
        "late_to_total_ratio": late_energy / max(total, 1e-20),
        "rt60_t10_estimate_ms": rt60_t10_ms,
        "rt60_t20_estimate_ms": rt60_t20_ms,
        "band_response_40_300_db": band_energy_ratio_db(mono, sample_rate, 40.0, 300.0),
        "band_response_300_2000_db": band_energy_ratio_db(mono, sample_rate, 300.0, 2000.0),
        "band_response_2000_8000_db": band_energy_ratio_db(mono, sample_rate, 2000.0, min(8000.0, sample_rate / 2.0)),
    }


def impulse_metrics(probe: np.ndarray, processed: np.ndarray, sample_rate: int, impulse_time_s: float) -> dict[str, Any]:
    impulse_sample = int(round(sample_rate * impulse_time_s))
    y = np.asarray(processed, dtype=np.float64)
    energy = y * y
    total = float(np.sum(energy))
    peak_sample = int(np.argmax(np.abs(y))) if y.size else 0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    early_stop = min(y.size, impulse_sample + int(round(0.05 * sample_rate)))
    late_start = min(y.size, impulse_sample + int(round(0.05 * sample_rate)))
    late_200_stop = min(y.size, impulse_sample + int(round(0.20 * sample_rate)))
    post = energy[impulse_sample:]
    cumulative = np.cumsum(post)
    if cumulative.size and cumulative[-1] > 0.0:
        t50 = int(np.searchsorted(cumulative, 0.50 * cumulative[-1])) / sample_rate * 1000.0
        t90 = int(np.searchsorted(cumulative, 0.90 * cumulative[-1])) / sample_rate * 1000.0
    else:
        t50 = float("nan")
        t90 = float("nan")
    return {
        "impulse_peak": peak,
        "impulse_peak_sample": peak_sample,
        "impulse_peak_time_ms": peak_sample / sample_rate * 1000.0,
        "impulse_peak_delay_from_input_ms": (peak_sample - impulse_sample) / sample_rate * 1000.0,
        "impulse_total_energy": total,
        "impulse_first_50ms_energy": float(np.sum(energy[impulse_sample:early_stop])),
        "impulse_50_200ms_energy": float(np.sum(energy[late_start:late_200_stop])),
        "impulse_late_after_50ms_ratio": float(np.sum(energy[late_start:]) / max(total, 1e-20)),
        "impulse_energy_t50_ms": float(t50),
        "impulse_energy_t90_ms": float(t90),
    }


def plot_rir_summary(mono_rir: np.ndarray, sample_rate: int, path: Path, title: str) -> None:
    if plt is None:
        return
    x = np.asarray(mono_rir, dtype=np.float64)
    t = np.arange(x.size) / float(sample_rate) * 1000.0
    energy = x * x
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(np.maximum(edc, 1e-20) / max(float(edc[0]) if edc.size else 0.0, 1e-20))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), dpi=140, sharex=True)
    axes[0].plot(t, x, linewidth=0.8)
    axes[0].set_ylabel("Mono RIR")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, edc_db, linewidth=0.9)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("EDC (dB)")
    axes[1].set_ylim(-80, 2)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_impulse_waveform(processed: np.ndarray, sample_rate: int, impulse_time_s: float, path: Path, title: str) -> None:
    if plt is None:
        return
    y = np.asarray(processed, dtype=np.float64)
    start = max(0, int(round((impulse_time_s - 0.02) * sample_rate)))
    stop = min(y.size, int(round((impulse_time_s + 0.25) * sample_rate)))
    t = np.arange(start, stop) / float(sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    ax.plot(t, y[start:stop], linewidth=0.8)
    ax.axvline(impulse_time_s, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_by_scene(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        "distance_m",
        "direct_to_residual_db",
        "c50_db",
        "late_to_total_ratio",
        "rt60_t10_estimate_ms",
        "rt60_t20_estimate_ms",
        "band_response_40_300_db",
        "band_response_300_2000_db",
        "band_response_2000_8000_db",
        "impulse_peak_delay_from_input_ms",
        "impulse_late_after_50ms_ratio",
        "impulse_energy_t90_ms",
    ]
    scenes = sorted({str(row["scene_type"]) for row in rows})
    out: list[dict[str, Any]] = []
    for scene in scenes:
        subset = [row for row in rows if row["scene_type"] == scene]
        item: dict[str, Any] = {"scene_type": scene, "count": len(subset)}
        item["los_count"] = sum(1 for row in subset if bool(row["is_los"]))
        item["nlos_count"] = len(subset) - int(item["los_count"])
        for key in numeric_keys:
            values = np.array([float(row[key]) for row in subset if row.get(key) not in ("", None) and np.isfinite(float(row[key]))])
            item[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{key}_std"] = float(np.std(values)) if values.size else float("nan")
        out.append(item)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_dir = args.input_dir.resolve()
    out = args.output_dir.resolve()
    audio_dir = out / "audio"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    for path in (audio_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(in_dir / "probe_manifest.csv")
    probe = make_impulse_probe(args.sample_rate, args.duration, args.impulse_time)
    dry_wav = audio_dir / "single_impulse_probe.wav"
    sf.write(dry_wav, probe, args.sample_rate)
    if not args.no_plots:
        save_stft_plot(probe, args.sample_rate, figure_dir / "single_impulse_probe_stft.png", "Single impulse probe STFT", args.stft_n_fft, args.stft_hop_length)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest):
        rir = np.load(in_dir / row["rir_path"])
        mono = mono_rir_from_foa(rir)
        processed = convolve_impulse_probe(probe, mono)
        case_id = row["case_id"]
        wav_path = audio_dir / f"{case_id}_single_impulse_mono.wav"
        sf.write(wav_path, processed, args.sample_rate)

        metrics = {
            **rir_metrics(rir, row, args.sample_rate, args.direct_window_ms, args.early_window_ms),
            **impulse_metrics(probe, processed, args.sample_rate, args.impulse_time),
            "rir_path": row["rir_path"],
            "impulse_mono_wav_path": str(wav_path.relative_to(out)),
        }
        rows.append(metrics)

        if not args.no_plots:
            plot_rir_summary(mono, args.sample_rate, figure_dir / f"{case_id}_rir_summary.png", f"{row['scene_type']} mono RIR")
            plot_impulse_waveform(processed, args.sample_rate, args.impulse_time, figure_dir / f"{case_id}_impulse_waveform.png", f"{row['scene_type']} single impulse output")
            save_stft_plot(processed, args.sample_rate, figure_dir / f"{case_id}_impulse_stft.png", f"{row['scene_type']} single impulse output STFT", args.stft_n_fft, args.stft_hop_length)
            save_delta_stft_plot(probe, processed, args.sample_rate, figure_dir / f"{case_id}_impulse_delta_stft.png", f"{row['scene_type']} impulse output-input STFT", args.stft_n_fft, args.stft_hop_length)

        print(json.dumps({"analyzed": index + 1, "case_id": case_id}, ensure_ascii=False))

    scene_summary = summarize_by_scene(rows)
    write_csv(out / "rir_impulse_metrics.csv", rows)
    write_json(out / "rir_impulse_metrics.json", rows)
    write_csv(out / "scene_type_summary.csv", scene_summary)
    write_json(out / "scene_type_summary.json", scene_summary)
    write_json(
        report_dir / "run_summary.json",
        {
            "input_dir": str(in_dir),
            "output_dir": str(out),
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "impulse_time": args.impulse_time,
            "dry_wav": str(dry_wav),
            "case_count": len(rows),
            "metrics_csv": str(out / "rir_impulse_metrics.csv"),
            "scene_type_summary_csv": str(out / "scene_type_summary.csv"),
            "notes": [
                "RIR analysis uses FOA W channel converted to mono as W * sqrt(2), matching backend.convolve_and_save.",
                "RT60 values are short-RIR decay estimates; treat NaN or unstable values as not available.",
                "The input probe is a 10-second mono signal with one impulse inside the first second.",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
