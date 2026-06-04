# SoundSpaces 材料语义链路修正说明

## 背景

在六场景 RIR 示例中，开启 SoundSpaces/RLR 材料后曾出现一个不合理现象：不同室内场景的 RIR 过于相似，且材料开启后间接反射长尾被异常截短。诊断发现这不是吸收、散射、空气 damping 或射线路径数量造成的，而是语义材料网格的 asset 格式问题。

## 根因

原先生成的 Habitat stage 使用 `*_semantic.ply` 作为 `semantic_asset`。SoundSpaces 能读取材料 JSON，也能加载语义类别，但在该 PLY semantic mesh 路径下，RLR 音频传播的间接反射会被异常截短。例如在关闭直达、绕射、透射，仅保留 indirect 的诊断中：

- PLY semantic asset：材料开启后 RIR 只有约 662 samples，50 ms 后能量为 0。
- OBJ semantic asset：材料开启后 RIR 恢复到约 29759 samples，50 ms 后有明显能量。

因此之前的问题不是材料参数本身错误，而是材料语义网格进入 RLR 的方式有问题。

## 修正内容

`src/soundspaces_adapter/semantic_stage.py` 现在默认生成带顶点语义颜色的 `*_semantic.obj`，并在 stage config 中使用它作为 `semantic_asset`。同时保留 PLY 输出作为诊断备用，但默认不再使用 PLY。

生成的 `.scn` 文件现在为每个语义对象写入：

- `class_`：材料/语义类别名，例如 `indoor_wall_reflective`
- `id`：与语义颜色对应的整数 ID
- `location`：该语义对象包围盒中心
- `size`：该语义对象包围盒尺寸

`location` 和 `size` 是 Habitat-Sim Gibson semantic scene 解析逻辑所使用的字段。

## 材料配置

材料系数来自 RLR/SoundSpaces 的 `mp3d_material_config.json` 对应材料项，写入到运行目录下的 `reports/occ_rlr_materials.json`。当前六类场景的合成表面采用如下手动分配：

- 室内地面：`Wood On Concrete`
- 室内墙面：`Gypsum Board`
- 室内天花板：`Gypsum Board`
- 固体遮挡物：`wood, Thick`
- 开阔草地地面：`Grass`
- 障碍森林地面：`Soil`
- 开放边界/天空吸收层：`Sound Proof`

注意：室内天花板使用 `Gypsum Board` 是按实验需求手动指定为“与墙面一致”。这仍使用官方材料库中的 `Gypsum Board` 系数，但不是把天花板映射到官方库里的 `Acoustic Tile`。

六场景示例和大规模数据生成现在默认开启 direct component 与 material transmission，使结果更接近真实材料传播；只有做遮挡消融或示意图时，才显式使用 `--disable-direct-for-nlos` 或 `--disable-transmission` 关闭对应机制。

## 重新运行命令

在项目根目录运行：

```bash
MPLCONFIGDIR=/tmp/occ_mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=src:. conda run -n occ_env python src/soundspaces_adapter/render_six_scene_impulse_probe.py \
  --output-dir generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_rays50000_figures \
  --indirect-ray-count 50000 \
  --direct-ray-count 500 \
  --source-ray-count 200 \
  --source-ray-depth 10 \
  --ir-duration 0.5 \
  --rir-plot-window-ms 120
```

## 本次验证结果

结果目录：

`generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_rays50000_figures`

检查到所有 stage config 均使用 `*_semantic.obj`：

```json
{
  "semantic_asset": "scene_xx_xxx_semantic.obj",
  "semantic_descriptor_filename": "scene_xx_xxx.scn"
}
```

`.scn` 中所有 `class_` 都能在 `occ_rlr_materials.json` 的 labels 中找到，没有缺失映射。

RIR 摘要：

```text
scene_01_baffle_room:      shape=(4, 26515), late50=0.517055
scene_02_l_shape_corridor: shape=(4, 28150), late50=0.399157
scene_03_t_shape_corridor: shape=(4, 28479), late50=0.300663
scene_04_empty_room:       shape=(4, 45056), late50=0.470684
scene_05_open_field:       shape=(4, 59142), late50≈0
scene_06_obstacle_forest:  shape=(4, 66072), late50=0.000193
```

这说明室内场景已经恢复明显长尾混响，开放场景保持很低的 late energy，符合开放/吸声边界预期。

## RIR 长尾可视化更新

材料语义链路修复后，室内 RIR 明显变长。为了避免只看前 120 ms 而误判混响尾部，六场景脚本现在默认使用 `--rir-plot-window-ms 500`。单场景 RIR 图包含三层：

- 单通道 RIR 线性幅值
- 单通道 RIR 幅值 dB
- Schroeder 能量衰减曲线

Schroeder 曲线比逐采样点幅值更适合判断混响是否衰减。逐采样点幅值会受大量离散反射、相位叠加和随机射线采样影响，局部峰值不一定平滑下降。

每个场景现在同时输出 1 张合并 RIR 图和 3 张独立图：

```text
*_rir_summary.png
*_rir_summary_waveform.png
*_rir_summary_amplitude_db.png
*_rir_summary_schroeder_decay.png
```

图内标注统一使用中文，不再使用 `mono` 作为坐标轴文字。非视距场景也会显示“几何直达延迟”，作为几何距离参考；这不是说存在可传播的直达声学路径，而是用于比较几何直线距离与观测首到达。三张独立 RIR 图的图例也会同时说明“几何直达延迟”和“观测首到达”的标记含义。

新版运行目录：

`generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_rays50000_figures_500ms_edc`

该版本新增总览图：

`figures/six_scene_schroeder_decay_overlay.png`

RT20/RT30 是分别用 -5~-25 dB 和 -5~-35 dB 衰减区间拟合并外推到 60 dB 的估计；RT60 使用 -5~-65 dB 衰减区间拟合。对应 RT 估计：

```text
scene_01_baffle_room:      RT20=0.459s, RT30=0.451s, RT60=0.443s
scene_02_l_shape_corridor: RT20=0.424s, RT30=0.419s, RT60=0.419s
scene_03_t_shape_corridor: RT20=0.457s, RT30=0.456s, RT60=0.455s
scene_04_empty_room:       RT20=0.601s, RT30=0.604s, RT60=0.604s
scene_05_open_field:       RT20=0.024s, RT30=0.018s, RT60=0.012s
scene_06_obstacle_forest:  RT20=0.110s, RT30=0.113s, RT60=0.111s
```

室内场景的 RT 偏长，符合当前“空房间 + Gypsum Board 墙/顶 + 硬质地面”的设置；开放场景和障碍森林的 RT 很短，符合开放/吸声边界。若需要更弱的室内混响，应调整表面材料分配，例如把天花板改回官方 `Acoustic Tile`，或使用更高吸收的室内表面。

## 推荐材料分配：天花板恢复 Acoustic Tile

在 semantic/material 链路修复后，重新将室内天花板从 `Gypsum Board` 改回官方 `Acoustic Tile`，地面保持 `Wood On Concrete`，墙面保持 `Gypsum Board`。这更接近最初的官方材料分配，也显著降低了空房间/走廊中过强的混响。

新版运行目录：

`generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc`

材料分配：

```text
室内地面: Wood On Concrete
室内墙面: Gypsum Board
室内天花板: Acoustic Tile
固体遮挡物: wood, Thick
开放边界: Sound Proof
```

与“墙/顶均为 Gypsum Board”的上一版相比：

```text
scene                  old RT30   new RT30   old late100   new late100
baffle_room              1.328      0.451       0.5917       0.1854
l_shape_corridor         1.409      0.419       0.6391       0.1929
t_shape_corridor         1.392      0.456       0.6583       0.2535
empty_room               2.195      0.604       0.4666       0.1202
open_field               0.018      0.018       0.0000       0.0000
obstacle_forest          0.113      0.113       0.0000       0.0000
```

这个版本的室内 RT 落在约 0.42-0.60 s，更适合作为当前六场景示例的推荐配置。

进一步修订后的图像输出目录：

`generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir`

加入独立图图例和 RT60 后的当前输出目录：

`generated_soundspaces_runs/six_scene_impulse_probe_material_path_fixed_ceiling_acoustic_tile_rays50000_500ms_edc_split_rir_rt60`

## 诊断命令

可用下面命令比较材料路径和非材料路径的 indirect-only 响应：

```bash
MPLCONFIGDIR=/tmp/occ_mpl NUMBA_DISABLE_JIT=1 PYTHONPATH=src:. conda run -n occ_env python src/soundspaces_adapter/diagnose_indirect_material_path.py \
  --output-dir generated_soundspaces_runs/indirect_only_material_path_check_obj_colored_normal_material \
  --scene-index 3 \
  --ray-count 50000 \
  --ir-duration 0.5
```

如需复现旧问题，可显式指定 `--semantic-asset-kind ply`。
