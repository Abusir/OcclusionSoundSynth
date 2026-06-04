#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MPLCONFIGDIR=/tmp/occ_mpl PYTHONPATH=src:. conda run -n occ_env python src/soundspaces_adapter/render_fire_sound_demo_from_rirs.py \
  --input-dir generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir_rt60 \
  --output-dir generated_soundspaces_runs/fire_sound_demo_esc50_logmel_zero_pad_from_rt60_rirs \
  --fire-audio git_version/examples/test_audio_bank/esc50_crackling_fire.wav \
  --duration 10 \
  --sample-rate 16000 \
  --mel-bins 80
