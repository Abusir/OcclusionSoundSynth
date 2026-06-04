#!/usr/bin/env bash
set -euo pipefail

MPLCONFIGDIR=/tmp/occ_mpl \
PYTHONPATH=src:. \
conda run -n occ_env python src/soundspaces_adapter/render_fire_sound_demo_from_rirs.py \
  --input-dir generated_soundspaces_runs/six_scene_impulse_probe_wall_ceiling_same_final_rays50000_figures \
  --output-dir generated_soundspaces_runs/fire_sound_demo_wall_ceiling_same_final_rays50000 \
  --fire-audio examples/generated/fire_sound_detection_compat/fire_sound_dataset_v2/SynthFireSound/fire/ss_example_000010_scene_04_empty_room_v08.wav \
  --duration 10.0 \
  --sample-rate 16000
