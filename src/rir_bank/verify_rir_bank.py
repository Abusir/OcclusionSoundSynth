from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rir_bank.manifest import read_manifest_csv
from rir_bank.validation import validate_manifest_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an OCC RIR bank manifest and saved RIR files.")
    parser.add_argument("--rir-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def write_invalid_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.rir_manifest.resolve()
    bank_dir = manifest_path.parent
    reports_dir = bank_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object]
    if not manifest_path.exists():
        report = {
            "passed": False,
            "errors": [f"missing_manifest:{manifest_path}"],
            "warnings": [],
        }
        (reports_dir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    try:
        rows = read_manifest_csv(manifest_path)
    except Exception as exc:
        report = {
            "passed": False,
            "errors": [f"manifest_read_failed:{type(exc).__name__}:{exc}"],
            "warnings": [],
        }
        (reports_dir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    report, invalid_rows, errors = validate_manifest_rows(rows, bank_dir)
    report["manifest_path"] = str(manifest_path)
    report["bank_dir"] = str(bank_dir)
    (reports_dir / "verification_report.json").write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    write_invalid_csv(reports_dir / "invalid_rirs.csv", invalid_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

