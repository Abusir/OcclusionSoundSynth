# Six Scene Wall-Ceiling Same Material, 50000 Rays

This experiment renders one impulse-probe RIR for each of the six OCC scene
types. Indoor ceilings use the same semantic material as indoor walls
(`indoor_wall_reflective`) instead of `indoor_ceiling_reflective`.

Run command:

```bash
bash git_version/experiments/six_scene_wall_ceiling_same_rays50000/run_six_scene_wall_ceiling_same_rays50000.sh
```

Key configuration:

- output: `generated_soundspaces_runs/six_scene_impulse_probe_wall_ceiling_same_final_rays50000_figures`
- sample rate: `16000`
- dry probe: 10 s with a single-sample impulse at 0.5 s
- RIR duration: `1.0` s
- indirect ray count: `50000`
- transmission: enabled
- direct component: kept for NLOS cases
- indoor ceiling material: `indoor_wall_reflective`
- outdoor boundary: absorbing shell with `sky_absorber`

Copied result:

- `result/` contains the generated run copied from
  `generated_soundspaces_runs/six_scene_impulse_probe_wall_ceiling_same_final_rays50000_figures`.
