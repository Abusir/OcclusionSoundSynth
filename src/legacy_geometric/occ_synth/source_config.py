from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    classes: tuple[str, ...] = ("fire", "others")
    dataset_name: str = "binary_fire_other"
    manifest_path: str | None = None

    @property
    def label_space(self) -> list[str]:
        return list(self.classes)


def _normalize_classes(classes: list[str]) -> tuple[str, ...]:
    cleaned = []
    seen = set()
    for cls in classes:
        value = str(cls).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    if not cleaned:
        raise ValueError("source class list is empty")
    return tuple(cleaned)


def _classes_from_json(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, list):
        values = []
        for item in payload:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                values.append(item.get("class") or item.get("label") or item.get("display_name") or item.get("name"))
        return _normalize_classes([value for value in values if value])
    if isinstance(payload, dict):
        if "classes" in payload:
            return _classes_from_json(payload["classes"])
        if "labels" in payload:
            return _classes_from_json(payload["labels"])
        if "ontology" in payload:
            return _classes_from_json(payload["ontology"])
    raise ValueError("JSON source manifest should contain a list, classes, labels, or ontology")


def _classes_from_csv(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            candidates = ["class", "label", "display_name", "name", "positive_labels"]
            field = next((name for name in candidates if name in reader.fieldnames), None)
            if field:
                values: list[str] = []
                for row in reader:
                    raw = row.get(field, "")
                    if field == "positive_labels":
                        values.extend(part.strip() for part in raw.split(","))
                    else:
                        values.append(raw)
                return _normalize_classes(values)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return _normalize_classes([row[0] for row in reader if row])


def load_source_config(
    source_classes: list[str] | None = None,
    manifest_path: Path | None = None,
    dataset_name: str | None = None,
) -> SourceConfig:
    if manifest_path is not None:
        path = manifest_path.resolve()
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            classes = _classes_from_json(payload)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            classes = _classes_from_csv(path)
        else:
            raise ValueError(f"Unsupported source manifest format: {path.suffix}")
        return SourceConfig(
            classes=classes,
            dataset_name=dataset_name or path.stem,
            manifest_path=str(path),
        )
    if source_classes:
        classes = _normalize_classes(source_classes)
        return SourceConfig(classes=classes, dataset_name=dataset_name or "custom")
    return SourceConfig(dataset_name=dataset_name or "binary_fire_other")
