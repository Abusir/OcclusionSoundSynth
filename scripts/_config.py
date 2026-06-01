from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def resolve_path(value: str | Path, *, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a scalar YAML value. Repeatable. Values are parsed as YAML scalars.",
    )


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        parsed = yaml.safe_load(value)
        merged[key.strip()] = parsed
    return merged


def bool_flag(name: str, enabled: bool) -> list[str]:
    return [name] if enabled else []


def append_option(args: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])
