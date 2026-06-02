from __future__ import annotations

import argparse
import sys

from _config import PROJECT_ROOT, SRC_ROOT, add_common_config_args, append_option, apply_overrides, bool_flag, load_yaml_config, resolve_path

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from run_rir_bank import main as rir_bank_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable SoundSpaces/Habitat-Sim FOA and mono RIR banks.")
    add_common_config_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = apply_overrides(load_yaml_config(config_path), args.set)
    cfg_dir = config_path.parent

    cli: list[str] = []
    append_option(cli, "--output-dir", resolve_path(cfg.get("output_dir", "../outputs/rir_bank"), config_dir=cfg_dir))
    append_option(cli, "--scenarios", cfg.get("scenarios", 2))
    append_option(cli, "--rirs-per-scenario", cfg.get("rirs_per_scenario"))
    append_option(cli, "--num-rirs", cfg.get("num_rirs", 2))
    append_option(cli, "--scene-sampling", cfg.get("scene_sampling", "stratified"))
    append_option(cli, "--sample-rate", cfg.get("sample_rate", 16000))
    append_option(cli, "--ir-duration", cfg.get("ir_duration", 0.2))
    append_option(cli, "--ray-count", cfg.get("ray_count", cfg.get("indirect_ray_count", 1000)))
    append_option(cli, "--direct-ray-count", cfg.get("direct_ray_count", 500))
    append_option(cli, "--indirect-ray-count", cfg.get("indirect_ray_count", cfg.get("ray_count", 1000)))
    append_option(cli, "--onset-threshold-db", cfg.get("onset_threshold_db", -80.0))
    append_option(cli, "--mono-scale", cfg.get("mono_scale"))
    append_option(cli, "--rir-format", cfg.get("rir_format", "both"))
    append_option(cli, "--seed", cfg.get("seed", 42))
    cli += bool_flag("--preserve-propagation-delay", bool(cfg.get("preserve_propagation_delay", False)))
    if not bool(cfg.get("enable_materials", True)):
        cli.append("--disable-materials")
    cli += bool_flag("--compute-metrics", bool(cfg.get("compute_metrics", True)))
    return int(rir_bank_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
