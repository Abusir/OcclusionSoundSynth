# Six-Scene Impulse Probe RT60 Run

This experiment stores the six-scene short impulse probe run generated with the corrected SoundSpaces semantic-material pipeline.

Material setup:

- indoor floor: Wood On Concrete
- indoor wall: Gypsum Board
- indoor ceiling: Acoustic Tile
- solid occluder: wood, Thick
- open boundary: Sound Proof absorber

The source is a 10 s probe with a single-sample impulse at 0.5 s. The run renders one scene for each of the six OCC structure types, with occluded source-receiver placement where geometric occlusion is possible.

Key outputs:

- manifest: `results/probe_manifest.csv`
- RIR arrays: `results/rirs/*_rir.npy`
- dry probe: `results/dry/short_impulse_probe.wav`
- RIR figures: `results/figures/*_rir_summary*.png`
- six-scene overlays: `results/figures/six_scene_rir_overlay.png`, `results/figures/six_scene_schroeder_decay_overlay.png`

The manifest includes `rir_rt20_s`, `rir_rt30_s`, and `rir_rt60_s`. Run `bash run.sh` from this directory to regenerate the result in `generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir_rt60`.
