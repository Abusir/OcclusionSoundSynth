from __future__ import annotations

import math
from typing import Iterable

import numpy as np


METRIC_NAMES = [
    "rt60",
    "edt",
    "early_late_ratio",
    "c50",
    "d50",
    "hf_loss",
    "peak_delay_ms",
    "rir_energy",
]


def _nan_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_NAMES}


def validate_mono_rir(rir: np.ndarray, *, silence_threshold: float = 1e-12) -> tuple[bool, str]:
    arr = np.asarray(rir, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return False, "empty_rir"
    if not np.all(np.isfinite(arr)):
        return False, "nan_or_inf"
    energy = float(np.sum(arr * arr))
    if energy <= silence_threshold:
        return False, "near_silent"
    return True, ""


def _safe_db_ratio(numerator: float, denominator: float, eps: float = 1e-20) -> float:
    return float(10.0 * math.log10(max(numerator, eps) / max(denominator, eps)))


def _decay_slope_rt60(rir: np.ndarray, sample_rate: int, start_db: float, stop_db: float) -> float:
    """Estimate RT60-like decay by fitting Schroeder EDC between two dB levels.

    The function returns ``nan`` when the requested decay range is not present.
    EDT uses 0..-10 dB extrapolated to -60 dB; RT60 here uses -5..-35 dB
    (T30-style) extrapolated to -60 dB. Short or noisy RIRs often lack a stable
    range, so callers should treat ``nan`` as an expected outcome.
    """

    arr = np.asarray(rir, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return float("nan")
    energy_rev = np.cumsum((arr[::-1] * arr[::-1]))[::-1]
    if energy_rev[0] <= 0.0:
        return float("nan")
    edc_db = 10.0 * np.log10(np.maximum(energy_rev / energy_rev[0], 1e-20))
    mask = (edc_db <= start_db) & (edc_db >= stop_db)
    indices = np.flatnonzero(mask)
    if indices.size < max(8, int(0.003 * sample_rate)):
        return float("nan")
    times = indices.astype(np.float64) / float(sample_rate)
    values = edc_db[indices]
    slope, _ = np.polyfit(times, values, 1)
    if slope >= -1e-9:
        return float("nan")
    return float(-60.0 / slope)


def _band_energy(power: np.ndarray, freqs: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def compute_rir_metrics(
    rir: np.ndarray,
    sample_rate: int,
    *,
    early_ms: float = 50.0,
) -> dict[str, float]:
    """Compute first-pass scalar metrics from a mono RIR.

    ``hf_loss`` is an intentionally simple spectral tilt proxy: it compares
    frequency-response energy in 0.2-2 kHz against 2-8 kHz and returns
    ``10*log10(low_band/high_band)``. Higher values therefore mean the RIR has
    relatively less high-frequency energy.
    """

    valid, _ = validate_mono_rir(rir)
    if not valid:
        return _nan_metrics()

    arr = np.asarray(rir, dtype=np.float64).reshape(-1)
    energy = float(np.sum(arr * arr))
    peak_index = int(np.argmax(np.abs(arr)))
    peak_delay_ms = float(1000.0 * peak_index / float(sample_rate))

    early_samples = max(1, min(arr.size, int(round(sample_rate * early_ms / 1000.0))))
    early_energy = float(np.sum(arr[:early_samples] ** 2))
    late_energy = float(np.sum(arr[early_samples:] ** 2))
    early_late_ratio = float(early_energy / late_energy) if late_energy > 0.0 else float("inf")
    c50 = _safe_db_ratio(early_energy, late_energy)
    d50 = float(early_energy / energy) if energy > 0.0 else float("nan")

    n_fft = int(2 ** math.ceil(math.log2(max(arr.size, 2))))
    spectrum = np.fft.rfft(arr, n=n_fft)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    low_energy = _band_energy(power, freqs, 200.0, 2000.0)
    high_hi = min(8000.0, float(sample_rate) / 2.0)
    high_energy = _band_energy(power, freqs, 2000.0, high_hi)
    hf_loss = _safe_db_ratio(low_energy, high_energy)

    return {
        "rt60": _decay_slope_rt60(arr, sample_rate, -5.0, -35.0),
        "edt": _decay_slope_rt60(arr, sample_rate, 0.0, -10.0),
        "early_late_ratio": early_late_ratio,
        "c50": c50,
        "d50": d50,
        "hf_loss": hf_loss,
        "peak_delay_ms": peak_delay_ms,
        "rir_energy": energy,
    }


def summarize_metric_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    rows = list(rows)
    payload["count"] = len(rows)
    payload["valid_count"] = sum(1 for row in rows if str(row.get("valid", "")).lower() in {"true", "1"})
    payload["invalid_count"] = len(rows) - int(payload["valid_count"])
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows if _is_finite_number(row.get(name))], dtype=np.float64)
        if values.size == 0:
            payload[name] = {"count": 0}
            continue
        payload[name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
    return payload


def _is_finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False

