from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MANIFEST_FIELDS = [
    "rir_id",
    "scene_id",
    "scenario_family",
    "split",
    "random_seed",
    "simulator_version",
    "sample_rate",
    "ir_duration",
    "ray_count",
    "direct_ray_count",
    "indirect_ray_count",
    "align_output_onset",
    "onset_threshold_db",
    "onset_sample",
    "mono_scale",
    "rir_path_foa",
    "rir_path_mono",
    "rir_format",
    "foa_channel_order",
    "foa_normalization",
    "foa_shape",
    "mono_derivation",
    "metric_channel",
    "source_position_x",
    "source_position_y",
    "source_position_z",
    "receiver_position_x",
    "receiver_position_y",
    "receiver_position_z",
    "receiver_yaw",
    "distance",
    "is_los",
    "is_occluded",
    "occlusion_type",
    "occlusion_severity",
    "rt60",
    "edt",
    "early_late_ratio",
    "c50",
    "d50",
    "hf_loss",
    "peak_delay_ms",
    "rir_energy",
    "valid",
    "invalid_reason",
]


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def read_manifest_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

