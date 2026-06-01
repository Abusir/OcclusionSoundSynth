from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


@dataclass(frozen=True)
class AudioItem:
    path: str
    label: str
    audio_id: str
    dataset: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AudioLibrary:
    items: tuple[AudioItem, ...]
    manifest_path: str
    dataset_name: str

    def __len__(self) -> int:
        return len(self.items)

    @property
    def classes(self) -> tuple[str, ...]:
        values = []
        seen = set()
        for item in self.items:
            if item.label not in seen:
                values.append(item.label)
                seen.add(item.label)
        return tuple(values)

    def sample(self, rng: random.Random) -> AudioItem:
        if not self.items:
            raise ValueError("audio library is empty")
        return rng.choice(self.items)

    def get(self, index: int) -> AudioItem:
        if not self.items:
            raise ValueError("audio library is empty")
        return self.items[index]


def _row_to_item(row: dict[str, Any], base_dir: Path, idx: int, dataset_name: str) -> AudioItem:
    path_value = row.get("path") or row.get("audio_path") or row.get("file") or row.get("filepath") or row.get("wav")
    if not path_value:
        raise ValueError(f"audio manifest row {idx} has no path/audio_path/file/filepath/wav field")
    label = row.get("label") or row.get("class") or row.get("source_type") or row.get("category") or "unknown"
    audio_id = row.get("audio_id") or row.get("id") or Path(str(path_value)).stem
    path = Path(str(path_value))
    if not path.is_absolute():
        path = base_dir / path
    metadata = {k: v for k, v in row.items() if k not in {"path", "audio_path", "file", "filepath", "wav", "label", "class", "source_type", "category", "audio_id", "id"}}
    return AudioItem(
        path=str(path.resolve()),
        label=str(label),
        audio_id=str(audio_id),
        dataset=str(row.get("dataset") or dataset_name),
        metadata=metadata,
    )


def load_audio_library(manifest_path: Path, dataset_name: str | None = None) -> AudioLibrary:
    path = manifest_path.resolve()
    dataset = dataset_name or path.stem
    base_dir = path.parent
    suffix = path.suffix.lower()
    items: list[AudioItem] = []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("audio") or payload.get("items") or payload.get("files") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON audio manifest should be a list or contain audio/items/files")
        for idx, row in enumerate(rows):
            if isinstance(row, str):
                row = {"path": row, "label": "unknown"}
            if not isinstance(row, dict):
                raise ValueError(f"audio manifest row {idx} should be a string or object")
            items.append(_row_to_item(row, base_dir, idx, dataset))
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for idx, row in enumerate(reader):
                items.append(_row_to_item(row, base_dir, idx, dataset))
    else:
        raise ValueError(f"Unsupported audio manifest format: {suffix}")
    if not items:
        raise ValueError(f"audio manifest is empty: {path}")
    missing = [item.path for item in items if not Path(item.path).exists()]
    if missing:
        raise FileNotFoundError(f"audio manifest contains missing files, first missing: {missing[0]}")
    return AudioLibrary(items=tuple(items), manifest_path=str(path), dataset_name=dataset)


def load_dry_audio(item: AudioItem, sample_rate: int, duration_s: float | None) -> np.ndarray:
    audio, input_sr = sf.read(item.path, always_2d=True)
    mono = np.mean(audio.astype(np.float32), axis=1)
    if input_sr != sample_rate:
        gcd = int(np.gcd(input_sr, sample_rate))
        mono = resample_poly(mono, sample_rate // gcd, input_sr // gcd).astype(np.float32)
    if mono.shape[0] == 0:
        raise ValueError(f"dry audio is empty: {item.path}")
    if duration_s is not None:
        target = int(round(sample_rate * duration_s))
        if mono.shape[0] < target:
            repeats = int(np.ceil(target / mono.shape[0]))
            mono = np.tile(mono, repeats)
        mono = mono[:target]
    mono = mono - float(np.mean(mono))
    peak = float(np.max(np.abs(mono)))
    if peak > 1e-9:
        mono = mono / peak
    return mono.astype(np.float32)
