# Six Scene Indoor All Gypsum Board Experiment

This experiment keeps the final six-scene SoundSpaces configuration, but overrides indoor surfaces and disables the direct component for NLOS cases:

- indoor floor: `indoor_wall_reflective`
- indoor wall: `indoor_wall_reflective`
- indoor ceiling: `indoor_wall_reflective`

`indoor_wall_reflective` is the RLR/SoundSpaces `Gypsum Board` material. Outdoor scenes keep their original grass/soil/sky materials.

For NLOS scenes, the SoundSpaces direct component is disabled. This avoids drawing or rendering an artificial straight-line direct arrival through an occluder. The RIR plots mark geometric direct delay only for LOS scenes; NLOS plots mark only the observed first arrival.

Run from the repository root:

```bash
bash git_version/experiments/six_scene_indoor_all_gypsum_rays50000/run_six_scene_indoor_all_gypsum_rays50000.sh
```

The fire-sound demo can be regenerated with:

```bash
bash git_version/experiments/six_scene_indoor_all_gypsum_rays50000/run_fire_sound_demo_indoor_all_gypsum.sh
```
