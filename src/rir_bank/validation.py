from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import METRIC_NAMES, summarize_metric_rows, validate_mono_rir


EXPECTED_FOA_ORDER = "ACN/SN3D [W,Y,Z,X]"
EXPECTED_MONO_DERIVATION = "foa_w_channel_sqrt2_aligned"
ACCEPTED_MONO_DERIVATIONS = {EXPECTED_MONO_DERIVATION, "foa_w_channel"}


def parse_shape(value: object) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        try:
            return [int(v) for v in value]
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except Exception:
        pass
    cleaned = text.strip("[]()")
    try:
        return [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    except Exception:
        return None


def load_npy(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32)


def validate_manifest_rows(rows: list[dict[str, Any]], base_dir: Path) -> tuple[dict[str, object], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    per_scene: dict[str, int] = {}
    sample_rates = set()
    ir_durations = set()
    ray_counts = set()

    for index, row in enumerate(rows):
        rir_id = str(row.get("rir_id") or f"row_{index}")
        scene_id = str(row.get("scene_id", ""))
        if scene_id:
            per_scene[scene_id] = per_scene.get(scene_id, 0) + 1
        sample_rates.add(str(row.get("sample_rate", "")))
        ir_durations.add(str(row.get("ir_duration", "")))
        ray_counts.add(str(row.get("ray_count", "")))

        row_errors: list[str] = []
        foa_path = _resolve_path(base_dir, str(row.get("rir_path_foa", "")))
        mono_path = _resolve_path(base_dir, str(row.get("rir_path_mono", "")))
        if not foa_path.exists():
            row_errors.append("missing_foa_path")
        if not mono_path.exists():
            row_errors.append("missing_mono_path")
        if str(row.get("foa_channel_order", "")) != EXPECTED_FOA_ORDER:
            row_errors.append("bad_foa_channel_order")
        if str(row.get("mono_derivation", "")) not in ACCEPTED_MONO_DERIVATIONS:
            row_errors.append("bad_mono_derivation")

        if foa_path.exists():
            try:
                foa = load_npy(foa_path)
                manifest_shape = parse_shape(row.get("foa_shape"))
                if manifest_shape is None or list(foa.shape) != manifest_shape:
                    row_errors.append("foa_shape_mismatch")
                if not np.all(np.isfinite(foa)):
                    row_errors.append("foa_nan_or_inf")
                if float(np.sum(foa * foa)) <= 1e-12:
                    row_errors.append("foa_near_silent")
            except Exception as exc:
                row_errors.append(f"foa_load_failed:{type(exc).__name__}")
        if mono_path.exists():
            try:
                mono = load_npy(mono_path).reshape(-1)
                valid, reason = validate_mono_rir(mono)
                if not valid:
                    row_errors.append(reason)
            except Exception as exc:
                row_errors.append(f"mono_load_failed:{type(exc).__name__}")

        manifest_valid = str(row.get("valid", "")).lower() in {"true", "1"}
        if row_errors:
            invalid = dict(row)
            invalid["verification_errors"] = ";".join(row_errors)
            invalid_rows.append(invalid)
            serious = [err for err in row_errors if err.startswith("missing_") or err.startswith("bad_") or "shape" in err]
            if serious:
                errors.append(f"{rir_id}: {';'.join(row_errors)}")
            else:
                warnings.append(f"{rir_id}: {';'.join(row_errors)}")
        elif not manifest_valid:
            invalid_rows.append(dict(row))
            warnings.append(f"{rir_id}: manifest marks invalid")

    if len(sample_rates) > 1:
        errors.append(f"inconsistent_sample_rate:{sorted(sample_rates)}")
    if len(ir_durations) > 1:
        errors.append(f"inconsistent_ir_duration:{sorted(ir_durations)}")
    if len(ray_counts) > 1:
        errors.append(f"inconsistent_ray_count:{sorted(ray_counts)}")

    report = {
        "manifest_rows": len(rows),
        "per_scene_counts": dict(sorted(per_scene.items())),
        "unique_sample_rates": sorted(sample_rates),
        "unique_ir_durations": sorted(ir_durations),
        "unique_ray_counts": sorted(ray_counts),
        "valid_count": sum(1 for row in rows if str(row.get("valid", "")).lower() in {"true", "1"}),
        "invalid_count": sum(1 for row in rows if str(row.get("valid", "")).lower() not in {"true", "1"}),
        "metric_summary": summarize_metric_rows(rows),
        "metric_names": METRIC_NAMES,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    return report, invalid_rows, errors


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path
