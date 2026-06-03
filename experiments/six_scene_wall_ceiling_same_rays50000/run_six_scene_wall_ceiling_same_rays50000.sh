#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

MPLCONFIGDIR=/tmp/occ_mpl \
NUMBA_DISABLE_JIT=1 \
PYTHONPATH=src:. \
conda run -n occ_env python src/soundspaces_adapter/render_six_scene_impulse_probe.py \
  --output-dir generated_soundspaces_runs/six_scene_impulse_probe_wall_ceiling_same_final_rays50000_figures \
  --ir-duration 1.0 \
  --indirect-ray-count 50000 \
  --allow-transmission \
  --keep-direct-for-nlos
