#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MPLCONFIGDIR=/tmp/occ_mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=src:. conda run -n occ_env python src/soundspaces_adapter/render_six_scene_impulse_probe.py \
  --output-dir generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir_rt60 \
  --indirect-ray-count 50000 \
  --direct-ray-count 500 \
  --source-ray-count 200 \
  --source-ray-depth 10 \
  --ir-duration 0.5
