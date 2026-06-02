# OCC SoundSpaces Material Assignment Table

The coefficients below are taken from the official RLR/SoundSpaces
`mp3d_material_config.json` distributed in `facebookresearch/rlr-audio-propagation`.
Frequency bands are 125, 250, 500, 1000, 2000, and 4000 Hz.

## Material Types

| Material ID | Source material in RLR config | Intended surface | Absorption | Scattering | Transmission |
|---|---|---|---:|---:|---:|
| `indoor_floor_hard` | `Wood On Concrete` | Indoor floors | 0.04 / 0.04 / 0.07 / 0.06 / 0.06 / 0.07 | 0.10 / 0.10 / 0.10 / 0.10 / 0.10 / 0.15 | 0.0040 / 0.0079 / 0.0056 / 0.0016 / 0.0014 / 0.0005 |
| `outdoor_ground_grass` | `Grass` | Open field ground | 0.11 / 0.26 / 0.60 / 0.69 / 0.92 / 0.99 | 0.30 / 0.30 / 0.40 / 0.50 / 0.60 / 0.70 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| `outdoor_ground_soil` | `Soil` | Obstacle forest ground | 0.15 / 0.25 / 0.40 / 0.55 / 0.60 / 0.60 | 0.10 / 0.20 / 0.25 / 0.40 / 0.55 / 0.70 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |
| `indoor_wall_reflective` | `Gypsum Board` | Indoor walls and corridor boundaries | 0.29 / 0.10 / 0.05 / 0.04 / 0.07 / 0.09 | 0.10 / 0.11 / 0.12 / 0.13 / 0.14 / 0.15 | 0.0350 / 0.0125 / 0.0056 / 0.0025 / 0.0013 / 0.0032 |
| `indoor_ceiling_reflective` | `Acoustic Tile` | Indoor ceilings | 0.50 / 0.70 / 0.60 / 0.70 / 0.70 / 0.50 | 0.10 / 0.15 / 0.20 / 0.20 / 0.25 / 0.30 | 0.050 / 0.040 / 0.030 / 0.020 / 0.005 / 0.002 |
| `solid_occluder` | `wood, Thick` | Baffles, obstacles, trunks, columns | 0.19 / 0.14 / 0.09 / 0.06 / 0.06 / 0.05 | 0.10 / 0.10 / 0.10 / 0.10 / 0.10 / 0.15 | 0.035 / 0.028 / 0.028 / 0.028 / 0.0011 / 0.0071 |
| `sky_absorber` | `Sound Proof` | Open scene boundary and open ceiling semantic sink | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |

## Scene Component Mapping

| Scene type | Floor | Wall / indoor boundary | Indoor ceiling | Obstacle | Open boundary | Open ceiling |
|---|---|---|---|---|---|---|
| `baffle_room` | `indoor_floor_hard` | `indoor_wall_reflective` | `indoor_ceiling_reflective` | `solid_occluder` | none | none |
| `l_shape_corridor` | `indoor_floor_hard` | `indoor_wall_reflective` | `indoor_ceiling_reflective` | none | none | none |
| `t_shape_corridor` | `indoor_floor_hard` | `indoor_wall_reflective` | `indoor_ceiling_reflective` | none | none | none |
| `empty_room` | `indoor_floor_hard` | `indoor_wall_reflective` | `indoor_ceiling_reflective` | none | none | none |
| `open_field` | `outdoor_ground_grass` | none | none | none | `sky_absorber` | `sky_absorber` |
| `obstacle_forest` | `outdoor_ground_soil` | none | none | `solid_occluder` | `sky_absorber` | `sky_absorber` |

In the current non-semantic OBJ SoundSpaces path, outdoor open boundary and open ceiling are not exported as acoustic geometry. They are still recorded in the assignment table so that a future semantic scene descriptor can map these components directly to `sky_absorber`.
