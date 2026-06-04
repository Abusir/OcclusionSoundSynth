from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SoundSpacesConfig:
    """Configuration shared by the adapter scripts.

    The defaults are conservative enough for debug runs. For final data
    generation, increase ``indirect_ray_count`` and run the convergence check.
    """

    sample_rate: int = 16000
    ir_duration_s: float = 1.0
    unit_scale: float = 1.0
    channel_type: str = "Ambisonics"
    channel_count: int = 4
    direct: bool = True
    indirect: bool = True
    diffraction: bool = True
    transmission: bool = True
    enable_materials: bool = True
    audio_materials_json: str | None = None
    semantic_material_stage: bool = True
    semantic_asset_kind: str = "obj_colored"
    frequency_bands: int = 4
    global_volume: float = 1.0
    direct_sh_order: int = 3
    indirect_sh_order: int = 1
    direct_ray_count: int = 500
    indirect_ray_count: int = 5000
    indirect_ray_depth: int = 200
    source_ray_count: int = 200
    source_ray_depth: int = 10
    max_diffraction_order: int = 10
    thread_count: int = 1
    output_directory: str = "generated_soundspaces_runs"
    write_ir_to_file: bool = False
    dump_wave_files: bool = False
    align_output_onset: bool = True
    onset_threshold_db: float = -80.0
    enable_rgb: bool = False
    enable_depth: bool = False
    visual_width: int = 320
    visual_height: int = 240
    visual_hfov_deg: float = 90.0
    visual_yaw_offset_deg: float = 0.0
    visual_pitch_deg: float = 0.0
    visual_sensor_height_offset_m: float = 0.0

    @property
    def ir_samples(self) -> int:
        return int(round(self.sample_rate * self.ir_duration_s))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
