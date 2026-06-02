from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import SoundSpacesConfig
from .coordinate import occ_to_habitat
from .material_database import write_occ_material_database
from .semantic_stage import write_semantic_stage_for_obj


class SoundSpacesUnavailableError(RuntimeError):
    """Raised when the optional SoundSpaces/Habitat-Sim stack is unavailable."""


def check_soundspaces_available() -> dict[str, object]:
    """Return import availability without importing the rest of the pipeline."""

    try:
        # The SoundSpaces 2.0 audio branch is old enough that importing
        # habitat_sim can abort in native code if torch is first imported from
        # Habitat's optional sensor-noise path. Importing torch explicitly
        # makes the initialization order deterministic.
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        import habitat_sim  # type: ignore

        return {
            "available": True,
            "habitat_sim_module": getattr(habitat_sim, "__file__", None),
            "has_audio_sensor_spec": hasattr(habitat_sim, "AudioSensorSpec"),
            "has_audio_config": hasattr(habitat_sim, "RLRAudioPropagationConfiguration"),
        }
    except Exception as exc:  # pragma: no cover - depends on local optional install
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


class SoundSpacesBackend:
    """Thin adapter around Habitat-Sim/SoundSpaces 2.0 audio APIs.

    The class uses guarded attribute access because SoundSpaces examples and
    Habitat-Sim builds expose slightly different Python names across versions.
    """

    def __init__(self, config: SoundSpacesConfig | None = None) -> None:
        self.config = config or SoundSpacesConfig()
        try:
            try:
                import torch  # noqa: F401
            except Exception:
                pass
            import habitat_sim  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SoundSpacesUnavailableError(
                "habitat_sim is not importable. Install SoundSpaces 2.0/Habitat-Sim before using the real backend."
            ) from exc
        self.habitat_sim = habitat_sim

    def _audio_materials_json(self, output_dir: Path) -> str | None:
        if not self.config.enable_materials:
            return None
        if self.config.audio_materials_json:
            return self.config.audio_materials_json
        path = output_dir / "occ_rlr_materials.json"
        write_occ_material_database(path)
        return str(path)

    def _make_acoustic_config(self) -> Any:
        hs = self.habitat_sim
        if not hasattr(hs, "RLRAudioPropagationConfiguration"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose RLRAudioPropagationConfiguration.")
        cfg = hs.RLRAudioPropagationConfiguration()
        mapping = {
            "sampleRate": self.config.sample_rate,
            "frequencyBands": self.config.frequency_bands,
            "globalVolume": self.config.global_volume,
            "directSHOrder": self.config.direct_sh_order,
            "indirectSHOrder": self.config.indirect_sh_order,
            "threadCount": self.config.thread_count,
            "maxIRLength": self.config.ir_duration_s,
            "unitScale": self.config.unit_scale,
            "directRayCount": self.config.direct_ray_count,
            "indirectRayCount": self.config.indirect_ray_count,
            "indirectRayDepth": self.config.indirect_ray_depth,
            "sourceRayCount": self.config.source_ray_count,
            "sourceRayDepth": self.config.source_ray_depth,
            "maxDiffractionOrder": self.config.max_diffraction_order,
            "direct": self.config.direct,
            "indirect": self.config.indirect,
            "diffraction": self.config.diffraction,
            "transmission": self.config.transmission,
            "writeIrToFile": self.config.write_ir_to_file,
            "dumpWaveFiles": self.config.dump_wave_files,
        }
        for key, value in mapping.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def _make_channel_layout(self) -> Any:
        hs = self.habitat_sim
        if not hasattr(hs, "RLRAudioPropagationChannelLayout"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose RLRAudioPropagationChannelLayout.")
        layout = hs.RLRAudioPropagationChannelLayout()
        if hasattr(layout, "channelCount"):
            layout.channelCount = self.config.channel_count
        if hasattr(layout, "type"):
            layout_type = getattr(hs, "RLRAudioPropagationChannelLayoutType", None)
            if layout_type is None and hasattr(hs, "sensor"):
                layout_type = getattr(hs.sensor, "RLRAudioPropagationChannelLayoutType", None)
            if layout_type is not None and hasattr(layout_type, self.config.channel_type):
                layout.type = getattr(layout_type, self.config.channel_type)
        return layout

    def build_audio_sensor_spec(
        self,
        receiver_occ_xyz: tuple[float, float, float] | None = None,
        output_dir: Path | None = None,
    ) -> Any:
        hs = self.habitat_sim
        if not hasattr(hs, "AudioSensorSpec"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose AudioSensorSpec.")
        spec = hs.AudioSensorSpec()
        spec.uuid = "audio_sensor"
        sensor_type = getattr(hs, "SensorType", None)
        if sensor_type is not None and hasattr(sensor_type, "AUDIO"):
            spec.sensor_type = sensor_type.AUDIO
        sensor_subtype = getattr(hs, "SensorSubType", None)
        if sensor_subtype is not None and hasattr(sensor_subtype, "IMPULSERESPONSE"):
            spec.sensor_subtype = sensor_subtype.IMPULSERESPONSE
        if hasattr(spec, "position"):
            # The listener follows the Habitat agent transform. Keep the audio
            # sensor as a zero-offset listener and set the agent state to the
            # receiver location in ``render_rir``.
            spec.position = [0.0, 0.0, 0.0]
        if hasattr(spec, "acousticsConfig"):
            spec.acousticsConfig = self._make_acoustic_config()
        if hasattr(spec, "channelLayout"):
            spec.channelLayout = self._make_channel_layout()
        if hasattr(spec, "enableMaterials"):
            spec.enableMaterials = self.config.enable_materials
        if output_dir is not None and hasattr(spec, "outputDirectory"):
            spec.outputDirectory = str(output_dir)
        return spec

    def build_visual_sensor_specs(self) -> list[Any]:
        hs = self.habitat_sim
        specs: list[Any] = []
        if not hasattr(hs, "CameraSensorSpec"):
            return specs
        sensor_type = getattr(hs, "SensorType", None)
        sensor_subtype = getattr(hs, "SensorSubType", None)
        if self.config.enable_rgb and sensor_type is not None:
            spec = hs.CameraSensorSpec()
            spec.uuid = "rgb_sensor"
            spec.sensor_type = sensor_type.COLOR
            spec.resolution = [self.config.visual_height, self.config.visual_width]
            spec.position = np.array([0.0, self.config.visual_sensor_height_offset_m, 0.0], dtype=np.float32)
            if hasattr(spec, "orientation"):
                spec.orientation = np.array([math.radians(self.config.visual_pitch_deg), 0.0, 0.0], dtype=np.float32)
            if sensor_subtype is not None:
                spec.sensor_subtype = sensor_subtype.PINHOLE
            if hasattr(spec, "hfov"):
                spec.hfov = float(self.config.visual_hfov_deg)
            specs.append(spec)
        if self.config.enable_depth and sensor_type is not None:
            spec = hs.CameraSensorSpec()
            spec.uuid = "depth_sensor"
            spec.sensor_type = sensor_type.DEPTH
            spec.resolution = [self.config.visual_height, self.config.visual_width]
            spec.position = np.zeros(3, dtype=np.float32)
            if hasattr(spec, "orientation"):
                spec.orientation = np.array([math.radians(self.config.visual_pitch_deg), 0.0, 0.0], dtype=np.float32)
            if sensor_subtype is not None:
                spec.sensor_subtype = sensor_subtype.PINHOLE
            if hasattr(spec, "hfov"):
                spec.hfov = float(self.config.visual_hfov_deg)
            specs.append(spec)
        return specs

    def _agent_rotation_toward(
        self,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
    ) -> Any:
        src = occ_to_habitat(source_occ_xyz).astype(np.float64)
        rec = occ_to_habitat(receiver_occ_xyz).astype(np.float64)
        direction = src - rec
        horizontal_norm = float(np.linalg.norm(direction[[0, 2]]))
        if horizontal_norm <= 1e-9:
            return None
        # Habitat cameras look along local -Z. Rotate around +Y so -Z points
        # toward the horizontal source direction.
        yaw = math.atan2(-direction[0], -direction[2]) + math.radians(self.config.visual_yaw_offset_deg)
        try:
            import quaternion  # type: ignore

            return quaternion.from_rotation_vector([0.0, yaw, 0.0])
        except Exception:
            return None

    def camera_forward_occ_xy(
        self,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Return the top-down OCC camera forward vector used by the adapter."""

        src = np.asarray(source_occ_xyz, dtype=np.float64)
        rec = np.asarray(receiver_occ_xyz, dtype=np.float64)
        direction = src[:2] - rec[:2]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return (1.0, 0.0)
        angle = math.atan2(float(direction[1]), float(direction[0])) + math.radians(self.config.visual_yaw_offset_deg)
        return (float(math.cos(angle)), float(math.sin(angle)))

    def _set_agent_pose(
        self,
        sim: Any,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
        infer_sensor_states: bool,
    ) -> Any:
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = occ_to_habitat(receiver_occ_xyz)
        rotation = self._agent_rotation_toward(source_occ_xyz, receiver_occ_xyz)
        if rotation is not None:
            state.rotation = rotation
        state.sensor_states = {}
        agent.set_state(state, infer_sensor_states)
        return agent

    def _add_audio_sensor(self, sim: Any, receiver_occ_xyz: tuple[float, float, float], output_dir: Path) -> Any:
        """Attach audio after simulator construction.

        The SoundSpaces/Habitat audio tutorial uses ``sim.add_sensor`` instead
        of placing ``AudioSensorSpec`` in ``AgentConfiguration``. Some 0.2.x
        audio builds crash during ``Simulator(...)`` when the audio spec is
        present at construction time, so the adapter follows the documented
        dynamic-add path.
        """

        spec = self.build_audio_sensor_spec(receiver_occ_xyz, output_dir)
        if not hasattr(sim, "add_sensor"):
            raise SoundSpacesUnavailableError("Simulator does not expose add_sensor for AudioSensorSpec.")
        sim.add_sensor(spec)
        return sim.get_agent(0)._sensors["audio_sensor"]

    def render_rir(
        self,
        scene_mesh_path: Path,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
        output_dir: Path,
    ) -> np.ndarray:
        """Render one SoundSpaces RIR.

        This method contains the integration shape used by the official docs:
        create a Habitat simulator, add an audio sensor, set the source
        transform, then query ``get_sensor_observations``. Exact simulator
        creation is version-specific; if the installed build does not expose the
        required API, the error message is explicit.
        """

        hs = self.habitat_sim
        if not hasattr(hs, "Simulator"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose Simulator.")
        if not scene_mesh_path.exists():
            raise FileNotFoundError(scene_mesh_path)
        audio_materials_json = self._audio_materials_json(output_dir)
        if audio_materials_json and scene_mesh_path.suffix.lower() == ".obj":
            scene_mesh_path = write_semantic_stage_for_obj(scene_mesh_path)

        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_mesh_path)
        if hasattr(sim_cfg, "create_renderer"):
            sim_cfg.create_renderer = False
        if hasattr(sim_cfg, "enable_physics"):
            sim_cfg.enable_physics = False
        if hasattr(sim_cfg, "scene_light_setup") and (self.config.enable_rgb or self.config.enable_depth):
            sim_cfg.scene_light_setup = ""
        if hasattr(sim_cfg, "override_scene_light_defaults") and (self.config.enable_rgb or self.config.enable_depth):
            sim_cfg.override_scene_light_defaults = True
        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [*self.build_visual_sensor_specs()]
        cfg = hs.Configuration(sim_cfg, [agent_cfg])
        sim = hs.Simulator(cfg)
        try:
            if self.config.enable_rgb or self.config.enable_depth:
                self._set_camera_light(sim)
            self._set_agent_pose(sim, source_occ_xyz, receiver_occ_xyz, False)
            audio_sensor = self._add_audio_sensor(sim, receiver_occ_xyz, output_dir)
            if audio_materials_json and hasattr(audio_sensor, "setAudioMaterialsJSON"):
                audio_sensor.setAudioMaterialsJSON(audio_materials_json)
            if hasattr(audio_sensor, "setAudioSourceTransform"):
                audio_sensor.setAudioSourceTransform(occ_to_habitat(source_occ_xyz))
            elif hasattr(audio_sensor, "set_audio_source_transform"):
                audio_sensor.set_audio_source_transform(occ_to_habitat(source_occ_xyz))
            else:
                raise SoundSpacesUnavailableError("Audio sensor does not expose a source-transform setter.")
            observations = sim.get_sensor_observations()
            rir = np.asarray(observations["audio_sensor"], dtype=np.float32)
            if not np.any(np.abs(rir) > 0.0):
                observations = sim.get_sensor_observations()
            rir = observations["audio_sensor"]
            return np.asarray(rir, dtype=np.float32)
        finally:
            if hasattr(sim, "close"):
                sim.close()

    def render_audio_visual(
        self,
        scene_mesh_path: Path,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
        output_dir: Path,
    ) -> dict[str, np.ndarray]:
        """Render audio plus optional RGB/depth observations from one agent.

        This uses the same source, receiver, simulator, and agent state for all
        sensors. It is the adapter entry point for audio-visual experiments.
        """

        hs = self.habitat_sim
        if not hasattr(hs, "Simulator"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose Simulator.")
        if not scene_mesh_path.exists():
            raise FileNotFoundError(scene_mesh_path)
        audio_materials_json = self._audio_materials_json(output_dir)
        if audio_materials_json and scene_mesh_path.suffix.lower() == ".obj":
            scene_mesh_path = write_semantic_stage_for_obj(scene_mesh_path)

        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_mesh_path)
        if hasattr(sim_cfg, "create_renderer"):
            sim_cfg.create_renderer = True
        if hasattr(sim_cfg, "enable_physics"):
            sim_cfg.enable_physics = False
        if hasattr(sim_cfg, "scene_light_setup") and (self.config.enable_rgb or self.config.enable_depth):
            sim_cfg.scene_light_setup = ""
        if hasattr(sim_cfg, "override_scene_light_defaults") and (self.config.enable_rgb or self.config.enable_depth):
            sim_cfg.override_scene_light_defaults = True
        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [*self.build_visual_sensor_specs()]
        cfg = hs.Configuration(sim_cfg, [agent_cfg])
        sim = hs.Simulator(cfg)
        try:
            if self.config.enable_rgb or self.config.enable_depth:
                self._set_camera_light(sim)
            self._set_agent_pose(sim, source_occ_xyz, receiver_occ_xyz, True)
            audio_sensor = self._add_audio_sensor(sim, receiver_occ_xyz, output_dir)
            if audio_materials_json and hasattr(audio_sensor, "setAudioMaterialsJSON"):
                audio_sensor.setAudioMaterialsJSON(audio_materials_json)
            if hasattr(audio_sensor, "setAudioSourceTransform"):
                audio_sensor.setAudioSourceTransform(occ_to_habitat(source_occ_xyz))
            elif hasattr(audio_sensor, "set_audio_source_transform"):
                audio_sensor.set_audio_source_transform(occ_to_habitat(source_occ_xyz))
            else:
                raise SoundSpacesUnavailableError("Audio sensor does not expose a source-transform setter.")
            observations = sim.get_sensor_observations()
            if "audio_sensor" in observations and not np.any(np.abs(observations["audio_sensor"]) > 0.0):
                observations = sim.get_sensor_observations()
            return {key: np.asarray(value) for key, value in observations.items()}
        finally:
            if hasattr(sim, "close"):
                sim.close()

    def render_visual_only(
        self,
        scene_mesh_path: Path,
        source_occ_xyz: tuple[float, float, float],
        receiver_occ_xyz: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        """Render RGB/depth only, without constructing the audio sensor."""

        hs = self.habitat_sim
        if not hasattr(hs, "Simulator"):
            raise SoundSpacesUnavailableError("This habitat_sim build does not expose Simulator.")
        if not scene_mesh_path.exists():
            raise FileNotFoundError(scene_mesh_path)

        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_mesh_path)
        if hasattr(sim_cfg, "create_renderer"):
            sim_cfg.create_renderer = True
        if hasattr(sim_cfg, "enable_physics"):
            sim_cfg.enable_physics = False
        if hasattr(sim_cfg, "scene_light_setup"):
            sim_cfg.scene_light_setup = ""
        if hasattr(sim_cfg, "override_scene_light_defaults"):
            sim_cfg.override_scene_light_defaults = True
        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [*self.build_visual_sensor_specs()]
        cfg = hs.Configuration(sim_cfg, [agent_cfg])
        sim = hs.Simulator(cfg)
        try:
            self._set_camera_light(sim)
            agent = sim.get_agent(0)
            state = agent.get_state()
            state.position = occ_to_habitat(receiver_occ_xyz)
            rotation = self._agent_rotation_toward(source_occ_xyz, receiver_occ_xyz)
            if rotation is not None:
                state.rotation = rotation
            state.sensor_states = {}
            agent.set_state(state, True)
            observations = sim.get_sensor_observations()
            return {key: np.asarray(value) for key, value in observations.items()}
        finally:
            if hasattr(sim, "close"):
                sim.close()

    def _set_camera_light(self, sim: Any) -> None:
        hs = self.habitat_sim
        try:
            light_info = hs.gfx.LightInfo
            light_model = hs.gfx.LightPositionModel
            setup = [
                light_info(vector=[0.0, 0.0, 1.0, 0.0], model=light_model.Camera),
                light_info(vector=[0.0, -1.0, 0.5, 0.0], model=light_model.Global),
                light_info(vector=[0.0, 1.0, 0.5, 0.0], model=light_model.Global),
            ]
            sim.set_light_setup(setup)
        except Exception:
            return

    def convolve_and_save(
        self,
        dry_audio: np.ndarray,
        rir: np.ndarray,
        foa_wav_path: Path,
        mono_wav_path: Path | None = None,
    ) -> dict[str, object]:
        dry = np.asarray(dry_audio, dtype=np.float32)
        arr = np.asarray(rir, dtype=np.float32)
        if dry.ndim != 1:
            raise ValueError("dry_audio must be mono")
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 2 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        onset_sample = self._rir_onset_sample(arr)
        rendered = np.zeros((dry.shape[0] + arr.shape[0] - 1, arr.shape[1]), dtype=np.float32)
        for channel in range(arr.shape[1]):
            rendered[:, channel] = np.convolve(dry, arr[:, channel], mode="full")
        if self.config.align_output_onset and onset_sample > 0:
            rendered = rendered[onset_sample:]
        rendered = rendered[: dry.shape[0]]
        if rendered.shape[0] < dry.shape[0]:
            pad = np.zeros((dry.shape[0] - rendered.shape[0], rendered.shape[1]), dtype=np.float32)
            rendered = np.concatenate([rendered, pad], axis=0)
        # SoundSpaces returns Ambisonics channels in Habitat's coordinate frame.
        # The project label convention is OCC-centric ACN/SN3D [W, Y, Z, X],
        # where OCC y is horizontal and OCC z is height. Because Habitat is
        # Y-up, the directional Y/Z channels must be swapped on export.
        if rendered.shape[1] == 4:
            rendered = rendered[:, [0, 2, 1, 3]]
        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.98:
            rendered *= 0.98 / peak
        foa_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(foa_wav_path, rendered, self.config.sample_rate)
        meta: dict[str, object] = {
            "backend": "soundspaces2",
            "config": asdict(self.config),
            "foa_wav_path": str(foa_wav_path),
            "channels": int(rendered.shape[1]),
            "num_samples": int(rendered.shape[0]),
            "channel_order": "ACN/SN3D [W, Y, Z, X]",
            "output_format": "FOA_ACN_SN3D_WYZX",
            "alignment": {
                "enabled": bool(self.config.align_output_onset),
                "method": "shift_left_by_rir_first_arrival",
                "onset_sample": int(onset_sample if self.config.align_output_onset else 0),
                "onset_seconds": float(onset_sample / self.config.sample_rate) if self.config.align_output_onset else 0.0,
                "threshold_db": float(self.config.onset_threshold_db),
            },
        }
        if rendered.shape[1] >= 1:
            mono = rendered[:, 0] * math.sqrt(2.0)
            mono_path = mono_wav_path or foa_wav_path.with_name(foa_wav_path.name.replace("_foa.wav", "_mono.wav"))
            sf.write(mono_path, mono, self.config.sample_rate)
            meta["mono_wav_path"] = str(mono_path)
            meta["mono_derivation"] = "mono = FOA_W * sqrt(2) from ACN/SN3D W channel"
        return meta

    def _rir_onset_sample(self, rir: np.ndarray) -> int:
        arr = np.asarray(rir, dtype=np.float32)
        if arr.ndim == 1:
            envelope = np.abs(arr)
        else:
            envelope = np.max(np.abs(arr), axis=1)
        if envelope.size == 0:
            return 0
        peak = float(np.max(envelope))
        if peak <= 0.0:
            return 0
        threshold = peak * (10.0 ** (float(self.config.onset_threshold_db) / 20.0))
        hits = np.flatnonzero(envelope >= threshold)
        return int(hits[0]) if hits.size else 0
