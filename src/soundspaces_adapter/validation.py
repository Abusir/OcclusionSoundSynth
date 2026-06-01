from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class RIRValidationResult:
    passed: bool
    distance_m: float
    expected_direct_delay_sample: int
    observed_first_peak_sample: int | None
    delay_error_samples: int | None
    early_energy: float
    late_energy: float
    direct_window_energy: float
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def expected_delay_sample(distance_m: float, sample_rate: int, speed_of_sound: float = SPEED_OF_SOUND) -> int:
    return int(round(float(distance_m) / speed_of_sound * sample_rate))


def energy_envelope(rir: np.ndarray) -> np.ndarray:
    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.ndim == 2 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.sqrt(np.sum(arr * arr, axis=1))


def first_peak_sample(envelope: np.ndarray, threshold_ratio: float = 0.08) -> int | None:
    if envelope.size == 0:
        return None
    peak = float(np.max(envelope))
    if peak <= 0.0:
        return None
    indices = np.flatnonzero(envelope >= peak * threshold_ratio)
    if indices.size == 0:
        return None
    return int(indices[0])


def validate_rir_physics(
    rir: np.ndarray,
    source_xyz: Iterable[float],
    receiver_xyz: Iterable[float],
    sample_rate: int,
    is_los: bool,
    tolerance_samples: int = 8,
) -> RIRValidationResult:
    source = np.asarray(list(source_xyz), dtype=np.float64)
    receiver = np.asarray(list(receiver_xyz), dtype=np.float64)
    distance = float(np.linalg.norm(source - receiver))
    expected = expected_delay_sample(distance, sample_rate)
    envelope = energy_envelope(rir)
    observed = first_peak_sample(envelope)
    delay_error = None if observed is None else int(observed - expected)
    direct_start = max(0, expected - tolerance_samples)
    direct_stop = min(envelope.shape[0], expected + tolerance_samples + 1)
    early_stop = min(envelope.shape[0], int(round(sample_rate * 0.05)))
    direct_energy = float(np.sum(envelope[direct_start:direct_stop] ** 2))
    early_energy = float(np.sum(envelope[:early_stop] ** 2))
    late_energy = float(np.sum(envelope[early_stop:] ** 2))

    notes: list[str] = []
    passed = True
    if observed is None:
        passed = False
        notes.append("RIR has no observable energy peak.")
    elif is_los and abs(delay_error or 0) > tolerance_samples:
        passed = False
        notes.append("LOS first peak is not close to the geometric direct-path delay.")
    if is_los and early_energy > 0.0 and direct_energy / early_energy < 0.02:
        passed = False
        notes.append("LOS direct-window energy is unexpectedly weak.")
    if not is_los:
        notes.append("NLOS samples need ablation checks; a later first peak is physically acceptable.")

    return RIRValidationResult(
        passed=passed,
        distance_m=distance,
        expected_direct_delay_sample=expected,
        observed_first_peak_sample=observed,
        delay_error_samples=delay_error,
        early_energy=early_energy,
        late_energy=late_energy,
        direct_window_energy=direct_energy,
        notes=notes,
    )


def estimate_foa_direction_acn_sn3d_wyzx(rir: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Estimate dominant FOA direction from an early RIR window.

    This assumes ACN/SN3D order ``[W, Y, Z, X]``. It is intended as a diagnostic
    for confirming a SoundSpaces build's channel convention, not as a universal
    ambisonic decoder.
    """

    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("expected an FOA RIR with at least four channels")
    window = arr[max(0, start) : min(arr.shape[0], stop)]
    if window.size == 0:
        return np.array([0.0, 0.0, 0.0])
    w = window[:, 0]
    y = float(np.dot(w, window[:, 1]))
    z = float(np.dot(w, window[:, 2]))
    x = float(np.dot(w, window[:, 3]))
    direction = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return direction / norm


def angular_error_deg(a_xyz: Iterable[float], b_xyz: Iterable[float]) -> float:
    a = np.asarray(list(a_xyz), dtype=np.float64)
    b = np.asarray(list(b_xyz), dtype=np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def write_validation_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
