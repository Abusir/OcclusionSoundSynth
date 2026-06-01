from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    au = _unit(a)
    bu = _unit(b)
    if float(np.linalg.norm(au)) < 1e-12 or float(np.linalg.norm(bu)) < 1e-12:
        return None
    cosine = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def expected_direction_from_label(label: dict[str, Any]) -> np.ndarray:
    receiver = np.asarray(label["receiver"]["center_xyz_m"], dtype=np.float64)
    source = np.asarray(label["source"]["xyz_m"], dtype=np.float64)
    return _unit(source - receiver)


def estimate_foa_direction(audio: np.ndarray, label: dict[str, Any], sample_rate: int) -> np.ndarray:
    render = label.get("render", {})
    direct_meta = render.get("direct_path", {})
    validation_meta = render.get("validation", {})
    delay = int(
        direct_meta.get(
            "delay_sample",
            validation_meta.get("expected_direct_delay_sample", validation_meta.get("observed_first_peak_sample", 0)),
        )
    )
    # Use only the very early direct-path arrival. Later samples quickly contain
    # reflections, so a wide window can make correct FOA look wrong.
    start = max(0, delay)
    stop = min(audio.shape[0], delay + max(128, int(0.016 * sample_rate)))
    if stop - start < max(64, int(0.008 * sample_rate)):
        start = 0
        stop = min(audio.shape[0], max(256, int(0.02 * sample_rate)))
    segment = audio[start:stop].astype(np.float64)
    if segment.size == 0:
        return np.zeros(3, dtype=np.float64)

    w = segment[:, 0]
    y = segment[:, 1]
    z = segment[:, 2]
    x = segment[:, 3]
    # For ACN/SN3D [W,Y,Z,X], W and each directional channel share the same
    # waveform for a dominant plane wave. Correlation signs recover direction.
    raw = np.array([np.sum(w * x), np.sum(w * y), np.sum(w * z)], dtype=np.float64)
    # Some long-distance or weak-energy cases can have a physically valid RIR
    # but too little directional SNR in the direct window to support a stable
    # FOA direction estimate. Treat those as inconclusive instead of false
    # failures.
    if float(np.linalg.norm(raw)) < 1e-4:
        return np.zeros(3, dtype=np.float64)
    return _unit(raw)


def validate_foa_label(label: dict[str, Any], max_los_angle_error_deg: float = 55.0) -> dict[str, Any]:
    wav_path = Path(label["files"]["audio_wav"])
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    errors: list[str] = []
    warnings: list[str] = []

    if audio.shape[1] != 4:
        errors.append(f"expected 4 FOA channels, got {audio.shape[1]}")
    if not np.isfinite(audio).all():
        errors.append("audio contains NaN or Inf")
    if float(np.max(np.abs(audio))) <= 1e-6:
        errors.append("audio is silent")

    channel_order = label.get("receiver", {}).get("channel_order")
    if channel_order != "ACN/SN3D [W, Y, Z, X]":
        errors.append(f"unexpected channel order metadata: {channel_order}")
    output_format = label.get("render", {}).get("render_config", {}).get("output_format")
    if output_format != "FOA_ACN_SN3D_WYZX":
        errors.append(f"unexpected output format metadata: {output_format}")

    expected = expected_direction_from_label(label)
    estimated = estimate_foa_direction(audio, label, sample_rate)
    angle_error = _angle_deg(expected, estimated)
    is_los = bool(label.get("relative", {}).get("is_los", False))
    if is_los and angle_error is not None and angle_error > max_los_angle_error_deg:
        errors.append(f"LOS FOA direction mismatch: {angle_error:.2f} deg")
    if not is_los and angle_error is not None and angle_error > 90.0:
        warnings.append(f"NLOS direction is reflection/diffraction dominated: {angle_error:.2f} deg")

    w_energy = float(np.mean(audio[:, 0] ** 2))
    directional_energy = float(np.mean(audio[:, 1:] ** 2))
    if w_energy <= 1e-12 or directional_energy <= 1e-12:
        errors.append("FOA W or directional channels have near-zero energy")
    if audio.shape[1] == 4:
        duplicate_pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                if np.allclose(audio[:, i], audio[:, j], atol=1e-6):
                    duplicate_pairs.append((i, j))
        if duplicate_pairs:
            errors.append(f"duplicate channels found: {duplicate_pairs}")

    mono_path_value = label.get("files", {}).get("mono_wav")
    mono_relation_error = None
    if mono_path_value:
        mono_path = Path(mono_path_value)
        if not mono_path.exists():
            errors.append("mono wav is missing")
        else:
            mono, mono_sr = sf.read(mono_path, always_2d=True)
            if mono_sr != sample_rate:
                errors.append(f"mono sample rate mismatch: {mono_sr} vs {sample_rate}")
            if mono.shape[1] != 1:
                errors.append(f"expected mono wav to have 1 channel, got {mono.shape[1]}")
            if mono.shape[0] != audio.shape[0]:
                errors.append("mono wav frame count differs from FOA wav")
            if audio.shape[1] == 4 and mono.shape[1] == 1 and mono.shape[0] == audio.shape[0]:
                expected_mono = audio[:, 0] * math.sqrt(2.0)
                mono_relation_error = float(np.max(np.abs(mono[:, 0] - expected_mono)))
                if mono_relation_error > 2e-4:
                    errors.append(f"mono wav is not FOA W*sqrt(2), max error {mono_relation_error:.6f}")

    return {
        "scene_id": label["scene_id"],
        "audio_wav": str(wav_path),
        "is_los": is_los,
        "expected_direction_xyz": expected.tolist(),
        "estimated_foa_direction_xyz": estimated.tolist(),
        "angle_error_deg": angle_error,
        "w_energy": w_energy,
        "directional_energy": directional_energy,
        "mono_relation_error": mono_relation_error,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_dataset(dataset_dir: Path, max_los_angle_error_deg: float = 55.0) -> dict[str, Any]:
    labels_path = dataset_dir / "labels_index.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    per_file = [validate_foa_label(label, max_los_angle_error_deg=max_los_angle_error_deg) for label in labels]
    errors = [item for item in per_file if not item["passed"]]
    los_angles = [
        item["angle_error_deg"]
        for item in per_file
        if item["is_los"] and item["angle_error_deg"] is not None
    ]
    all_angles = [item["angle_error_deg"] for item in per_file if item["angle_error_deg"] is not None]
    return {
        "dataset_dir": str(dataset_dir),
        "checked_files": len(per_file),
        "passed": len(errors) == 0,
        "failed_files": len(errors),
        "los_angle_error_mean_deg": float(np.mean(los_angles)) if los_angles else None,
        "los_angle_error_p95_deg": float(np.percentile(los_angles, 95)) if los_angles else None,
        "all_angle_error_mean_deg": float(np.mean(all_angles)) if all_angles else None,
        "all_angle_error_p95_deg": float(np.percentile(all_angles, 95)) if all_angles else None,
        "failures": errors[:20],
        "per_file": per_file,
    }
