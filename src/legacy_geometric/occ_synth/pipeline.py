from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

from shapely.geometry import mapping

from .acoustics import RenderConfig, render_foa_audio
from .audio_library import AudioItem, AudioLibrary, load_dry_audio
from .extrusion import export_scene_obj
from .sampling import AcousticPlacement, sample_placement
from .scene_generator import Scene2D, generate_all_scenes
from .source_config import SourceConfig
from .validation import summarize_validation, validate_catalog, validate_label, validate_obj
from .visualization import plot_catalog_overview, plot_scene, plot_waveform


def _round_floats(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, list):
        return [_round_floats(v, ndigits) for v in value]
    if isinstance(value, tuple):
        return [_round_floats(v, ndigits) for v in value]
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    return value


def scene_geometry_record(scene: Scene2D) -> dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "scene_index": scene.scene_index,
        "scene_type_id": scene.scene_type_id,
        "scene_type": scene.scene_type,
        "variant_index": scene.variant_index,
        "is_outdoor": scene.is_outdoor,
        "height_m": scene.height_m,
        "area_m2": scene.boundary.area,
        "walkable_area_m2": scene.walkable_area.area,
        "bounds_xy_m": list(scene.bounds),
        "params": scene.params,
        "boundary_geojson": mapping(scene.boundary),
        "obstacles_geojson": [mapping(obs) for obs in scene.obstacles],
    }


def make_label(
    example_id: str,
    scene: Scene2D,
    placement: AcousticPlacement,
    files: dict[str, str],
    render_meta: dict[str, Any],
    source_config: SourceConfig,
    audio_item: AudioItem | None = None,
) -> dict[str, Any]:
    audio_input = None
    if audio_item is not None:
        audio_input = {
            "audio_id": audio_item.audio_id,
            "path": audio_item.path,
            "label": audio_item.label,
            "dataset": audio_item.dataset,
            "metadata": audio_item.metadata or {},
        }
    return _round_floats(
        {
            "example_id": example_id,
            "scene_id": scene.scene_id,
            "scene_index": scene.scene_index,
            "room_case_id": f"type{scene.scene_type_id:02d}_variant{scene.variant_index:02d}",
            "scene_type_id": scene.scene_type_id,
            "scene_type": scene.scene_type,
            "is_outdoor": scene.is_outdoor,
            "room": {
                "height_m": scene.height_m,
                "area_m2": scene.boundary.area,
                "walkable_area_m2": scene.walkable_area.area,
                "bounds_xy_m": list(scene.bounds),
                "params": scene.params,
            },
            "receiver": {
                "format": "FOA",
                "center_xyz_m": placement.receiver_xyz,
                "channel_order": "ACN/SN3D [W, Y, Z, X]",
            },
            "source": {
                "type": placement.source_type,
                "class": placement.source_type,
                "dataset": audio_item.dataset if audio_item is not None else source_config.dataset_name,
                "label_space": source_config.label_space,
                "manifest_path": source_config.manifest_path,
                "xyz_m": placement.source_xyz,
            },
            "audio_input": audio_input,
            "occlusion": {
                "is_direct_path_obstructed": not placement.is_los,
                "direct_path_status": "obstructed" if not placement.is_los else "clear",
                "reason": "direct path intersects obstacle or wall" if not placement.is_los else "direct path is inside walkable free space",
                "obstruction_count": placement.obstruction_count,
                "obstruction_types": list(placement.obstruction_types),
            },
            "relative": {
                "azimuth_rad": placement.azimuth_rad,
                "azimuth_deg": placement.azimuth_rad * 180.0 / 3.141592653589793,
                "azimuth_reference": {
                    "zero_direction": "+x axis",
                    "positive_direction": "counter_clockwise toward +y",
                    "plane": "xy",
                },
                "elevation_rad": placement.elevation_rad,
                "distance_m": placement.distance_m,
                "is_los": placement.is_los,
                "is_obstructed": not placement.is_los,
                "obstruction_count": placement.obstruction_count,
            },
            "files": files,
            "render": render_meta,
        }
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    output_dir: Path,
    variants_per_type: int = 10,
    seed: int = 20260415,
    smoke_limit: int | None = None,
    render_config: RenderConfig | None = None,
    source_config: SourceConfig | None = None,
    audio_library: AudioLibrary | None = None,
    num_examples: int | None = None,
    scene_sampling: str = "sequential",
    visualize_all: bool = True,
) -> dict[str, Any]:
    render_config = render_config or RenderConfig()
    source_config = source_config or SourceConfig()
    output_dir = output_dir.resolve()
    geometry_dir = output_dir / "geometry"
    audio_dir = output_dir / "audio"
    label_dir = output_dir / "labels"
    figure_dir = output_dir / "figures"
    validation_errors: list[str] = []

    scenes = generate_all_scenes(variants_per_type=variants_per_type, seed=seed)
    validation_errors.extend(validate_catalog(scenes, variants_per_type))
    all_scene_count = len(scenes)
    if smoke_limit is not None:
        scenes = scenes[:smoke_limit]
    if audio_library is not None:
        source_config = SourceConfig(
            classes=audio_library.classes,
            dataset_name=audio_library.dataset_name,
            manifest_path=audio_library.manifest_path,
        )
    if scene_sampling not in {"sequential", "random"}:
        raise ValueError("scene_sampling must be sequential or random")

    if visualize_all:
        plot_catalog_overview(generate_all_scenes(variants_per_type=variants_per_type, seed=seed), figure_dir / "catalog_overview.png")

    rng = random.Random(seed + 1009)
    labels: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    scene_lookup = {scene.scene_id: scene for scene in scenes}
    for scene in scenes:
        geometry_records.append(_round_floats(scene_geometry_record(scene)))
        files = export_scene_obj(scene, geometry_dir)
        validation_errors.extend(validate_obj(Path(files["obj"])))

    example_count = num_examples if num_examples is not None else len(scenes)
    if example_count <= 0:
        raise ValueError("num_examples must be positive")

    for idx in range(example_count):
        if scene_sampling == "random":
            scene = rng.choice(scenes)
        else:
            scene = scenes[idx % len(scenes)]
        scene_files = {
            "obj": str(geometry_dir / f"{scene.scene_id}.obj"),
            "mtl": str(geometry_dir / f"{scene.scene_id}.mtl"),
        }
        audio_item = audio_library.sample(rng) if audio_library is not None else None
        source_types = [audio_item.label] if audio_item is not None else source_config.label_space

        prefer_obstructed = scene.scene_type_id in {1, 2, 3, 6} and rng.random() < 0.65
        placement = sample_placement(
            scene,
            rng,
            source_types=source_types,
            prefer_obstructed=prefer_obstructed,
        )
        example_id = f"example_{idx:06d}_{scene.scene_id}"
        dry_audio = load_dry_audio(audio_item, render_config.sample_rate, render_config.duration_s) if audio_item is not None else None
        wav_path = audio_dir / f"{example_id}_foa.wav"
        mono_wav_path = audio_dir / f"{example_id}_mono.wav"
        render_meta = render_foa_audio(
            scene,
            placement,
            wav_path,
            render_config,
            seed + idx * 37,
            dry_audio=dry_audio,
            output_mono_wav=mono_wav_path,
        )
        files = dict(scene_files)
        files["audio_wav"] = str(wav_path)
        files["mono_wav"] = str(mono_wav_path)

        scene_plot = figure_dir / f"{example_id}_layout.png"
        plot_scene(scene, placement, scene_plot)
        files["layout_png"] = str(scene_plot)

        if idx < 6:
            wave_plot = figure_dir / f"{example_id}_waveform.png"
            plot_waveform(wav_path, wave_plot)
            files["waveform_png"] = str(wave_plot)

        label = make_label(example_id, scene, placement, files, render_meta, source_config, audio_item=audio_item)
        label_path = label_dir / f"{example_id}.json"
        label["files"]["label_json"] = str(label_path)
        write_json(label_path, label)
        validation_errors.extend(validate_label(label, scene_lookup[label["scene_id"]], allowed_source_classes=set(source_config.classes)))
        labels.append(label)

    write_json(output_dir / "scene_geometry_catalog.json", geometry_records)
    write_json(output_dir / "labels_index.json", labels)

    summary = {
        "seed": seed,
        "variants_per_type": variants_per_type,
        "generated_scene_total": all_scene_count,
        "available_scene_total": len(scenes),
        "rendered_scene_total": len(labels),
        "scene_sampling": scene_sampling,
        "num_examples": example_count,
        "output_dir": str(output_dir),
        "render_config": asdict(render_config),
        "source_config": {
            "classes": source_config.label_space,
            "dataset_name": source_config.dataset_name,
            "manifest_path": source_config.manifest_path,
        },
        "audio_library": {
            "enabled": audio_library is not None,
            "manifest_path": audio_library.manifest_path if audio_library is not None else None,
            "num_items": len(audio_library.items) if audio_library is not None else 0,
        },
        "validation": summarize_validation(validation_errors),
        "directories": {
            "geometry": str(geometry_dir),
            "audio": str(audio_dir),
            "labels": str(label_dir),
            "figures": str(figure_dir),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
