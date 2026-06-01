from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math

import numpy as np
import soundfile as sf

from .sampling import AcousticPlacement
from .scene_generator import Scene2D


SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class RenderConfig:
    sample_rate: int = 16000
    duration_s: float = 3.0
    rir_duration_s: float = 1.0
    reflection_order: int = 2
    output_format: str = "FOA_ACN_SN3D_WYZX"
    backend: str = "geometric_foa_v1"


def _unit_direction(receiver: np.ndarray, source: np.ndarray) -> np.ndarray:
    vec = source - receiver
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return vec / norm


def _foa_encode_mono(signal: np.ndarray, direction_xyz: np.ndarray) -> np.ndarray:
    # ACN/SN3D channel order for first order ambisonics: W, Y, Z, X.
    x, y, z = direction_xyz
    channels = np.zeros((signal.shape[0], 4), dtype=np.float32)
    channels[:, 0] = signal * (1.0 / math.sqrt(2.0))
    channels[:, 1] = signal * y
    channels[:, 2] = signal * z
    channels[:, 3] = signal * x
    return channels


def synthesize_dry_sound(source_type: str, sample_rate: int, duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    envelope = np.linspace(1.0, 0.85, n)
    if source_type == "fire":
        low = rng.normal(0.0, 0.35, n)
        dry = np.cumsum(low)
        dry = dry - np.mean(dry)
        dry = dry / (np.max(np.abs(dry)) + 1e-9)
        crackle = rng.normal(0.0, 1.0, n) * (rng.random(n) < 0.010)
        for lag, gain in [(80, 0.55), (211, 0.35), (377, 0.25)]:
            crackle[lag:] += crackle[:-lag] * gain
        dry += 0.35 * crackle
    elif source_type == "others":
        carrier = np.sin(2 * np.pi * 1650.0 * t) + 0.35 * np.sin(2 * np.pi * 890.0 * t)
        gate = (np.sin(2 * np.pi * 1.6 * t) > -0.35).astype(np.float64)
        dry = carrier * gate + 0.04 * rng.normal(0.0, 1.0, n)
    elif source_type == "fire_alarm":
        carrier = np.sin(2 * np.pi * 3050.0 * t) + 0.5 * np.sin(2 * np.pi * 3550.0 * t)
        gate = (np.sin(2 * np.pi * 2.2 * t) > -0.15).astype(np.float64)
        dry = carrier * gate
    elif source_type == "smoke_detector":
        dry = np.sin(2 * np.pi * 4200.0 * t) * (np.sin(2 * np.pi * 3.0 * t) > 0.2)
    elif source_type == "crackle":
        impulses = rng.random(n) < 0.012
        dry = rng.normal(0.0, 1.0, n) * impulses
        for lag, gain in [(80, 0.55), (211, 0.35), (377, 0.25)]:
            dry[lag:] += dry[:-lag] * gain
    else:
        low = rng.normal(0.0, 0.35, n)
        dry = np.cumsum(low)
        dry = dry - np.mean(dry)
        dry = dry / (np.max(np.abs(dry)) + 1e-9)
        dry += 0.15 * rng.normal(0.0, 1.0, n)
    dry *= envelope
    dry = dry / (np.max(np.abs(dry)) + 1e-9)
    return dry.astype(np.float32)


def _add_path(
    rir: np.ndarray,
    receiver: np.ndarray,
    source: np.ndarray,
    gain: float,
    sample_rate: int,
) -> None:
    distance = float(np.linalg.norm(source - receiver))
    delay = int(round(distance / SPEED_OF_SOUND * sample_rate))
    if delay >= rir.shape[0]:
        return
    attenuation = gain / max(distance, 0.35)
    direction = _unit_direction(receiver, source)
    encoded = _foa_encode_mono(np.array([attenuation], dtype=np.float32), direction)[0]
    rir[delay, :] += encoded


def render_foa_audio(
    scene: Scene2D,
    placement: AcousticPlacement,
    output_wav: Path,
    config: RenderConfig,
    seed: int,
    dry_audio: np.ndarray | None = None,
    output_mono_wav: Path | None = None,
) -> dict[str, object]:
    receiver = np.array(placement.receiver_xyz, dtype=np.float64)
    source = np.array(placement.source_xyz, dtype=np.float64)
    rir_len = int(config.sample_rate * config.rir_duration_s)
    rir = np.zeros((rir_len, 4), dtype=np.float32)

    direct_gain = 1.0 if placement.is_los else 0.18 / (1.0 + placement.obstruction_count)
    direct_direction = _unit_direction(receiver, source)
    direct_distance = float(np.linalg.norm(source - receiver))
    direct_delay_sample = int(round(direct_distance / SPEED_OF_SOUND * config.sample_rate))
    _add_path(rir, receiver, source, direct_gain, config.sample_rate)

    minx, miny, maxx, maxy = scene.boundary.bounds
    room_gain = 0.42 if not scene.is_outdoor else 0.12
    image_sources = [
        np.array([2 * minx - source[0], source[1], source[2]]),
        np.array([2 * maxx - source[0], source[1], source[2]]),
        np.array([source[0], 2 * miny - source[1], source[2]]),
        np.array([source[0], 2 * maxy - source[1], source[2]]),
        np.array([source[0], source[1], -source[2]]),
        np.array([source[0], source[1], 2 * scene.height_m - source[2]]),
    ]
    for idx, image in enumerate(image_sources):
        _add_path(rir, receiver, image, room_gain * (0.88 ** idx), config.sample_rate)

    if not placement.is_los:
        midpoint = (source + receiver) / 2.0
        midpoint[2] = max(source[2], receiver[2]) + 0.45
        _add_path(rir, receiver, midpoint, 0.28 / (1.0 + placement.obstruction_count), config.sample_rate)

    tail_start = int(0.045 * config.sample_rate)
    if tail_start < rir_len:
        rng = np.random.default_rng(seed + 7919)
        decay_seconds = 0.42 if not scene.is_outdoor else 0.16
        tail_t = np.arange(rir_len - tail_start) / config.sample_rate
        tail = rng.normal(0.0, 1.0, (rir_len - tail_start, 4)).astype(np.float32)
        tail *= np.exp(-tail_t[:, None] / decay_seconds).astype(np.float32)
        rir[tail_start:] += tail * (0.010 if not scene.is_outdoor else 0.003)

    dry = dry_audio
    dry_source = "manifest_audio" if dry is not None else "synthetic"
    if dry is None:
        dry = synthesize_dry_sound(placement.source_type, config.sample_rate, config.duration_s, seed)
    if dry.ndim != 1:
        raise ValueError("dry_audio must be mono")
    target = int(round(config.sample_rate * config.duration_s))
    if dry.shape[0] != target:
        raise ValueError(f"dry_audio length mismatch: got {dry.shape[0]}, expected {target}")
    rendered = np.zeros((dry.shape[0] + rir.shape[0] - 1, 4), dtype=np.float32)
    for channel in range(4):
        rendered[:, channel] = np.convolve(dry, rir[:, channel], mode="full")
    rendered = rendered[: dry.shape[0]]
    mono = rendered[:, 0] * math.sqrt(2.0)
    peak = float(max(np.max(np.abs(rendered)), np.max(np.abs(mono))))
    if peak > 0.98:
        rendered *= 0.98 / peak
        mono = rendered[:, 0] * math.sqrt(2.0)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, rendered, config.sample_rate)
    mono_path = output_mono_wav or output_wav.with_name(output_wav.name.replace("_foa.wav", "_mono.wav"))
    mono_peak = float(np.max(np.abs(mono)))
    mono_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mono_path, mono.astype(np.float32), config.sample_rate)
    return {
        "wav_path": str(output_wav),
        "mono_wav_path": str(mono_path),
        "sample_rate": config.sample_rate,
        "num_channels": 4,
        "mono_num_channels": 1,
        "num_samples": int(rendered.shape[0]),
        "peak": float(np.max(np.abs(rendered))),
        "mono_peak": float(np.max(np.abs(mono))),
        "mono_derivation": "mono = FOA_W * sqrt(2) from ACN/SN3D W channel",
        "dry_source": dry_source,
        "rir_nonzero_samples": int(np.count_nonzero(np.abs(rir).sum(axis=1) > 0.0)),
        "direct_path": {
            "distance_m": direct_distance,
            "delay_sample": direct_delay_sample,
            "gain": direct_gain,
            "direction_xyz": direct_direction.tolist(),
            "channel_order": "ACN/SN3D [W, Y, Z, X]",
            "encoding": "W=s/sqrt(2), Y=s*y, Z=s*z, X=s*x",
        },
        "render_config": asdict(config),
    }
