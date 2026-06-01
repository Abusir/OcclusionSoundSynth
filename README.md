# OCC Data Synth

OCC Data Synth is a SoundSpaces/Habitat-Sim based toolkit for acoustic scene simulation and data synthesis. It exports two complementary workflows:

- Direct audio synthesis: input mono WAV files, render SoundSpaces/Habitat-Sim room impulse responses, then save reverberant/occluded FOA and mono audio.
- Reusable RIR bank generation: pre-render FOA/mono RIR arrays so large audio corpora can be convolved later without repeatedly launching Habitat-Sim.

The project also includes a reproducible 3D scene viewer for checking generated meshes, source/receiver positions, direct paths, and camera footprints.

<p align="center">
  <img src="docs/assets/scene_3d_screenshot.png" alt="Example 3D acoustic scene" width="86%">
</p>

## Highlights

- SoundSpaces/Habitat-Sim is the acoustic backend; the legacy geometric code is used for scene generation and visualization, not as a simplified replacement.
- FOA output is exported in ACN/SN3D channel order `[W, Y, Z, X]`.
- The RIR bank path separates expensive acoustic simulation from cheap large-scale convolution.
- The browser viewer loads the generated OBJ scene and checks receiver/source placement before scaling up a run.

## Example Scene Checks

The included figures are generated from the same procedural scene and placement code used by the SoundSpaces/Habitat-Sim runs. Blue marks the FOA receiver, red marks the source, and the line shows the direct acoustic path.

| Baffle room | L-shaped corridor |
| --- | --- |
| <img src="docs/assets/scene_00_layout.png" alt="Baffle room layout" width="100%"> | <img src="docs/assets/scene_12_layout.png" alt="L-shaped corridor layout" width="100%"> |
| Empty room | Obstacle forest |
| <img src="docs/assets/scene_34_layout.png" alt="Empty room layout" width="100%"> | <img src="docs/assets/scene_55_layout.png" alt="Obstacle forest layout" width="100%"> |

## Repository Layout

```text
configs/                 Example YAML configs
docs/assets/             README figures
examples/test_audio_bank Small mono-audio smoke-test manifest
scripts/                 Public command-line entry points
src/legacy_geometric/    Programmatic scene geometry, sampling, plots
src/soundspaces_adapter/ SoundSpaces/Habitat-Sim adapter and validation
src/rir_bank/            RIR manifest, metrics, and validation helpers
viewer3d/                Browser-based Three.js scene viewer
```

Generated outputs are written under `outputs/` by default and are ignored by git.

## Environment

The verified local environment is `/home/digao/miniconda3/envs/occ_env` with Python 3.10, `habitat-sim==0.2.2`, `habitat==0.2.2`, `sound-spaces==0.1.1`, and CUDA 11.8 PyTorch.

To create a similar base environment:

```bash
conda env create -f environment.yml
conda activate occ_env
```

Then install the SoundSpaces/Habitat stack following the SoundSpaces 2.0 audio branch instructions. These packages are version-sensitive and may require source/editable installs; keep the versions above when reproducing the tested setup.

For the viewer:

```bash
cd viewer3d
npm install
```

## Data Paths

All public scripts accept a YAML config:

```bash
python scripts/synthesize_audio.py --config configs/audio_synthesis.yaml
```

Paths inside configs are resolved relative to the config file. You can override scalar values without editing YAML:

```bash
python scripts/generate_rirs.py --config configs/rir_generation.yaml --set num_rirs=10
```

Audio manifests may be CSV/TSV/JSON. CSV rows should contain at least `path,label,audio_id`; relative audio paths are resolved against the manifest file.

## Direct Audio Synthesis

Smoke-test command:

```bash
PYTHONPATH=src python scripts/synthesize_audio.py --config configs/audio_synthesis.yaml
```

Main outputs:

- `audio/*_foa.wav`: FOA audio, ACN/SN3D channel order `[W, Y, Z, X]`.
- `audio/*_mono.wav`: mono derived from FOA W as `W * sqrt(2)`.
- `labels/*.json`: source/receiver positions, occlusion metadata, render config.
- `geometry/*.obj`: generated Habitat mesh.
- `figures/*_layout.png`: top-down scene check.

Increase `num_examples`, `variants_per_type`, and ray counts in `configs/audio_synthesis.yaml` for medium-scale generation.

## Batch RIR Generation

Smoke-test command:

```bash
PYTHONPATH=src python scripts/generate_rirs.py --config configs/rir_generation.yaml
```

Main outputs:

- `rirs/scene_*/rir_*_foa.npy`: FOA RIR shaped `[4, T]`.
- `rirs/scene_*/rir_*_mono.npy`: mono RIR derived from FOA W.
- `rir_manifest.csv` and `rir_manifest.jsonl`: reusable metadata for later convolution.
- `reports/verification_report.json`: shape and value validation.

The RIR workflow is intentionally independent of dry-audio convolution, so the saved `.npy` files can be reused with large external audio datasets.

## 3D Scene Visualization

Generate viewer assets:

```bash
PYTHONPATH=src python scripts/visualize_scene.py --config configs/visualization.yaml
```

This writes OBJ/MTL geometry and `viewer3d/data/latest_report.json`. Start the viewer:

```bash
cd viewer3d
npm run serve
```

Open `http://127.0.0.1:8765/`. The viewer shows the 3D mesh, receiver, source, direct path, and camera footprint. Set `render_habitat_rgb: true` in `configs/visualization.yaml` to also request Habitat RGB/depth observations when the local Habitat renderer is available.

## Common Issues

- `habitat_sim is not importable`: install the SoundSpaces/Habitat-Sim stack in the active conda environment.
- Native crash on simulator construction: confirm the SoundSpaces audio branch and Habitat-Sim versions match the verified stack.
- Empty or invalid RIR: try larger `indirect_ray_count`, check the exported OBJ in `viewer3d`, and inspect `reports/invalid_rirs.csv`.
- Viewer cannot load `three`: run `npm install` inside `viewer3d`.
- Viewer cannot load mesh: regenerate assets with `scripts/visualize_scene.py` and serve from `viewer3d`, not by opening the HTML file directly.
