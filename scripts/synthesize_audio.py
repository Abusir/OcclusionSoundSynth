from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _config import PROJECT_ROOT, SRC_ROOT, add_common_config_args, append_option, apply_overrides, bool_flag, load_yaml_config, resolve_path

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from soundspaces_adapter.build_dataset import main as build_dataset_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize occluded/reverberant audio with SoundSpaces/Habitat-Sim.")
    add_common_config_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = apply_overrides(load_yaml_config(config_path), args.set)
    cfg_dir = config_path.parent

    cli: list[str] = []
    append_option(cli, "--output-dir", resolve_path(cfg.get("output_dir", "../outputs/audio_synthesis"), config_dir=cfg_dir))
    append_option(cli, "--audio-manifest", resolve_path(cfg["audio_manifest"], config_dir=cfg_dir))
    append_option(cli, "--source-dataset-name", cfg.get("source_dataset_name", "example_audio"))
    append_option(cli, "--variants-per-type", cfg.get("variants_per_type", 2))
    append_option(cli, "--seed", cfg.get("seed", 20260416))
    append_option(cli, "--num-examples", cfg.get("num_examples", 1))
    append_option(cli, "--scene-sampling", cfg.get("scene_sampling", "random"))
    append_option(cli, "--audio-sampling", cfg.get("audio_sampling", "cover_once_then_random"))
    append_option(cli, "--audio-start-index", cfg.get("audio_start_index", 0))
    append_option(cli, "--example-start-index", cfg.get("example_start_index", 0))
    append_option(cli, "--sample-rate", cfg.get("sample_rate", 16000))
    append_option(cli, "--duration", cfg.get("duration", "source"))
    append_option(cli, "--ir-duration", cfg.get("ir_duration", 0.2))
    append_option(cli, "--ray-count", cfg.get("ray_count", cfg.get("indirect_ray_count", 1000)))
    append_option(cli, "--thread-count", cfg.get("thread_count", 1))
    append_option(cli, "--max-attempts", cfg.get("max_attempts", 160))
    append_option(cli, "--onset-threshold-db", cfg.get("onset_threshold_db", -80.0))
    cli += bool_flag("--preserve-propagation-delay", bool(cfg.get("preserve_propagation_delay", False)))
    if not bool(cfg.get("enable_materials", True)):
        cli.append("--disable-materials")
    if bool(cfg.get("no_overview", False)):
        cli.append("--no-overview")
    if bool(cfg.get("no_progress_bar", True)):
        cli.append("--no-progress-bar")
    if bool(cfg.get("resume", False)):
        cli.append("--resume")
    return int(build_dataset_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
