from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soundspaces_adapter.backend import check_soundspaces_available
from soundspaces_adapter.coordinate import assert_round_trip, distance_m, occ_to_habitat
from soundspaces_adapter.validation import expected_delay_sample, validate_rir_physics
from soundspaces_adapter.visualize_debug import plot_rir_debug


def main() -> int:
    output_dir = ROOT / "soundspaces_adapter" / "generated" / "adapter_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    receiver = (1.0, 1.0, 1.4)
    source = (4.2, 2.1, 1.2)
    assert_round_trip(receiver)
    assert_round_trip(source)
    dist_occ = distance_m(receiver, source)
    dist_hab = distance_m(occ_to_habitat(receiver), occ_to_habitat(source))
    if abs(dist_occ - dist_hab) > 1e-6:
        raise AssertionError("coordinate transform changed Euclidean distance")

    sample_rate = 16000
    rir = np.zeros((sample_rate, 4), dtype=np.float32)
    delay = expected_delay_sample(dist_occ, sample_rate)
    direction = (np.asarray(source) - np.asarray(receiver)) / dist_occ
    x, y, z = direction
    rir[delay, 0] = 1.0 / np.sqrt(2.0)
    rir[delay, 1] = y
    rir[delay, 2] = z
    rir[delay, 3] = x
    rir[delay + 53, :] += np.array([0.10, 0.03, 0.00, -0.04], dtype=np.float32)
    validation = validate_rir_physics(rir, source, receiver, sample_rate, is_los=True)
    plot_rir_debug(rir, source, receiver, sample_rate, output_dir / "synthetic_los_rir.png")

    report = {
        "passed": validation.passed,
        "coordinate_distance_preserved": True,
        "soundspaces_available": check_soundspaces_available(),
        "synthetic_los_validation": validation.to_dict(),
        "figure": str(output_dir / "synthetic_los_rir.png"),
    }
    (output_dir / "adapter_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if validation.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
