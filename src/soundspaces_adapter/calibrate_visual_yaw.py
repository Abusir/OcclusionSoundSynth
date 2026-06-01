from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a yaw sweep to calibrate Habitat RGB camera footprint.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yaw_calibration"))
    parser.add_argument("--scene-index", type=int, default=3)
    parser.add_argument("--case", choices=["los", "nlos"], default="los")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--ray-count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--visual-pitch", type=float, default=0.0)
    parser.add_argument("--yaw-offsets", type=str, default="0,90,180,270")
    return parser.parse_args()


def run_one(args: argparse.Namespace, yaw: int, root: Path) -> dict[str, object]:
    out = root / f"yaw_{yaw}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("verify_visual_only.py")),
        "--output-dir",
        str(out),
        "--scene-index",
        str(args.scene_index),
        "--case",
        args.case,
        "--sample-rate",
        str(args.sample_rate),
        "--ir-duration",
        str(args.ir_duration),
        "--ray-count",
        str(args.ray_count),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-pitch",
        str(args.visual_pitch),
        "--visual-yaw-offset",
        str(yaw),
    ]
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    report_path = out / "reports" / "visual_only_report.json"
    report: dict[str, object]
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"status": "missing_report", "passed": False}
    report["yaw_offset_deg"] = yaw
    report["returncode"] = completed.returncode
    if completed.returncode != 0:
        report["stderr_tail"] = completed.stderr[-2000:]
    return report


def make_contact_sheet(root: Path, reports: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(len(reports), 2, figsize=(9, max(3.0, 3.0 * len(reports))), dpi=130)
    if len(reports) == 1:
        axes = axes[None, :]
    for row, report in enumerate(reports):
        yaw = report["yaw_offset_deg"]
        cam_path = Path(str(report.get("camera_geometry_figure", "")))
        rgb_path = Path(str(report.get("rgb", {}).get("path", ""))) if isinstance(report.get("rgb"), dict) else Path()
        if cam_path.exists():
            axes[row, 0].imshow(imageio.imread(cam_path))
        axes[row, 0].set_title(f"yaw {yaw}: footprint")
        if rgb_path.exists():
            axes[row, 1].imshow(imageio.imread(rgb_path))
        axes[row, 1].set_title(f"yaw {yaw}: RGB, passed={report.get('passed')}")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    out = root / "yaw_calibration_contact.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    yaws = [int(item.strip()) for item in args.yaw_offsets.split(",") if item.strip()]
    reports = [run_one(args, yaw, root) for yaw in yaws]
    contact = make_contact_sheet(root, reports)
    summary = {
        "contact_sheet": str(contact),
        "yaw_offsets": yaws,
        "reports": reports,
        "note": (
            "Use this sheet to choose visual_yaw_offset_deg for the current Habitat-Sim build. "
            "The footprint is a top-down diagnostic, not a pixel-exact RGB projection."
        ),
    }
    summary_path = root / "yaw_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
