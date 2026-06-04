from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import soundfile as sf

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None
    FontProperties = None
    Line2D = None
    Patch = None
    Poly3DCollection = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy_geometric.occ_synth.extrusion import export_scene_obj
from legacy_geometric.occ_synth.sampling import AcousticPlacement, sample_placement
from legacy_geometric.occ_synth.scene_generator import Scene2D, generate_all_scenes
from soundspaces_adapter.analyze_rir_impulse_probe import mono_rir_from_foa
from soundspaces_adapter.backend import SoundSpacesBackend, check_soundspaces_available
from soundspaces_adapter.config import SoundSpacesConfig
from soundspaces_adapter.material_database import (
    SCENE_MATERIAL_ASSIGNMENTS,
    scene_material_assignment,
    set_material_damping_scale,
    set_material_medium,
    write_occ_material_database,
    write_scene_material_assignments,
)
from soundspaces_adapter.render_flat_spectrum_probe import (
    save_delta_stft_plot,
    save_spectrum_plot,
    save_stft_plot,
    save_transfer_plot,
    spectrum_summary,
)
from soundspaces_adapter.validation import validate_rir_physics


SCENE_ORDER = [
    "baffle_room",
    "l_shape_corridor",
    "t_shape_corridor",
    "empty_room",
    "open_field",
    "obstacle_forest",
]

SCENE_TITLES = {
    "baffle_room": "挡板房间",
    "l_shape_corridor": "L 形走廊",
    "t_shape_corridor": "T 形走廊",
    "empty_room": "空房间",
    "open_field": "开阔场地",
    "obstacle_forest": "障碍物林",
}

TARGET_LOS = {
    "baffle_room": False,
    "l_shape_corridor": False,
    "t_shape_corridor": False,
    "empty_room": True,
    "open_field": True,
    "obstacle_forest": False,
}

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"),
]
FONT_PATH = next((path for path in FONT_CANDIDATES if path.exists()), None)
FONT = FontProperties(fname=str(FONT_PATH)) if FontProperties is not None and FONT_PATH is not None else None

if plt is not None and FONT is not None:
    plt.rcParams.update(
        {
            "font.family": FONT.get_name(),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _font_kwargs() -> dict[str, Any]:
    return {"fontproperties": FONT} if FONT is not None else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one correctly-materialed SoundSpaces RIR per OCC scene type using "
            "a 10-second short-impulse probe."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_soundspaces_runs/six_scene_impulse_probe_correct_materials"),
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--variants-per-type", type=int, default=10)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--impulse-time", type=float, default=0.5)
    parser.add_argument(
        "--impulse-duration",
        type=float,
        default=0.0,
        help=(
            "Short probe duration in seconds. Use 0 for a single-sample delta "
            "probe with a flat 0-Nyquist spectrum; nonzero values must be <= 0.1."
        ),
    )
    parser.add_argument("--ir-duration", type=float, default=0.2)
    parser.add_argument("--direct-ray-count", type=int, default=500)
    parser.add_argument("--indirect-ray-count", type=int, default=1000)
    parser.add_argument("--indirect-ray-depth", type=int, default=200)
    parser.add_argument("--source-ray-count", type=int, default=200)
    parser.add_argument("--source-ray-depth", type=int, default=10)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument(
        "--los-control-distance",
        type=float,
        default=6.0,
        help="Target source-receiver distance for LOS control scenes, in meters. Use <= 0 to disable.",
    )
    parser.add_argument(
        "--distance-tolerance",
        type=float,
        default=0.35,
        help="Accepted distance tolerance in meters for target-distance sampling.",
    )
    parser.add_argument(
        "--allow-transmission",
        action="store_true",
        default=True,
        help=(
            "Allow acoustic transmission through materials. This is enabled by default; "
            "the flag is kept for compatibility with older run scripts."
        ),
    )
    parser.add_argument(
        "--disable-transmission",
        action="store_true",
        help="Disable acoustic transmission for ablation or schematic occlusion-only examples.",
    )
    parser.add_argument(
        "--disable-materials",
        action="store_true",
        help="Disable the semantic/RLR material database and use the SoundSpaces default material path.",
    )
    parser.add_argument(
        "--outdoor-boundary-mode",
        choices=("auto", "absorbing_shell", "geometric_open"),
        default="auto",
        help=(
            "Outdoor geometry export mode. auto uses absorbing side/top faces with "
            "semantic materials enabled and true geometric open boundaries when "
            "materials are disabled."
        ),
    )
    parser.add_argument(
        "--keep-direct-for-nlos",
        action="store_true",
        default=True,
        help=(
            "Keep the direct acoustic component enabled for NLOS cases. This is enabled "
            "by default for physically realistic material simulations."
        ),
    )
    parser.add_argument(
        "--disable-direct-for-nlos",
        action="store_true",
        help="Disable the direct component for NLOS cases for schematic occlusion-only examples.",
    )
    parser.add_argument(
        "--align-output-onset",
        action="store_true",
        help="Align convolved output onset, matching the legacy flat-spectrum run.",
    )
    parser.add_argument(
        "--indoor-surfaces-gypsum",
        action="store_true",
        help=(
            "Use the RLR/SoundSpaces Gypsum Board material for indoor floors, "
            "walls, and ceilings. Outdoor scenes keep their existing materials."
        ),
    )
    parser.add_argument(
        "--indoor-balanced-reflective",
        action="store_true",
        help=(
            "Use a synthetic middle-ground reflective material for indoor floors, "
            "walls, and ceilings. Outdoor scenes keep their existing materials."
        ),
    )
    parser.add_argument(
        "--material-damping-scale",
        type=float,
        default=1.0,
        help="Scale the damping curve written to the semantic material JSON. Use 0 to disable material damping.",
    )
    parser.add_argument(
        "--material-medium-density",
        type=float,
        default=None,
        help="Override density written to every semantic material. Useful for testing air-like vs legacy RLR medium values.",
    )
    parser.add_argument(
        "--material-medium-speed",
        type=float,
        default=None,
        help="Override speed written to every semantic material. Useful for testing air-like vs legacy RLR medium values.",
    )
    parser.add_argument("--onset-threshold-db", type=float, default=-80.0)
    parser.add_argument(
        "--rir-plot-window-ms",
        type=float,
        default=500.0,
        help="Time window shown in RIR figures, in milliseconds. Use <= 0 to show the full RIR.",
    )
    parser.add_argument("--stft-n-fft", type=int, default=1024)
    parser.add_argument("--stft-hop-length", type=int, default=256)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def apply_indoor_surfaces_gypsum_override() -> None:
    for scene_type in ("baffle_room", "l_shape_corridor", "t_shape_corridor", "empty_room"):
        assignment = SCENE_MATERIAL_ASSIGNMENTS[scene_type]
        assignment["floor"] = "indoor_wall_reflective"
        assignment["wall"] = "indoor_wall_reflective"
        assignment["ceiling"] = "indoor_wall_reflective"


def apply_indoor_balanced_reflective_override() -> None:
    for scene_type in ("baffle_room", "l_shape_corridor", "t_shape_corridor", "empty_room"):
        assignment = SCENE_MATERIAL_ASSIGNMENTS[scene_type]
        assignment["floor"] = "indoor_balanced_reflective"
        assignment["wall"] = "indoor_balanced_reflective"
        assignment["ceiling"] = "indoor_balanced_reflective"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_short_impulse_probe(
    sample_rate: int,
    duration_s: float,
    impulse_time_s: float,
    impulse_duration_s: float,
) -> np.ndarray:
    if duration_s <= 0.0:
        raise ValueError("--duration must be positive")
    if not 0.0 <= impulse_duration_s <= 0.1:
        raise ValueError("--impulse-duration must be >= 0 and <= 0.1 seconds")
    n = int(round(sample_rate * duration_s))
    start = int(round(sample_rate * impulse_time_s))
    width = 1 if impulse_duration_s == 0.0 else max(1, int(round(sample_rate * impulse_duration_s)))
    if start < 0 or start + width > n:
        raise ValueError("--impulse-time plus --impulse-duration must fit inside --duration")
    audio = np.zeros(n, dtype=np.float32)
    if width == 1:
        pulse = np.ones(1, dtype=np.float32)
    else:
        pulse = np.hanning(width).astype(np.float32)
        if not np.any(pulse):
            pulse[:] = 1.0
    peak = float(np.max(np.abs(pulse)))
    audio[start : start + width] = 0.95 * pulse / max(peak, 1e-12)
    return audio


def select_scene_and_placement(
    scene_type: str,
    scenes: list[Scene2D],
    rng: random.Random,
    target_distance_m: float | None = None,
    distance_tolerance_m: float = 0.35,
) -> tuple[Scene2D, AcousticPlacement, int]:
    candidates = [scene for scene in scenes if scene.scene_type == scene_type]
    rng.shuffle(candidates)
    want_los = TARGET_LOS[scene_type]
    attempts = 0
    best_near_distance: tuple[float, Scene2D, AcousticPlacement, int] | None = None
    for scene in candidates:
        for _ in range(2500):
            attempts += 1
            placement = sample_placement(
                scene,
                rng,
                source_types=["short_impulse_probe"],
                min_distance_m=1.25,
                prefer_obstructed=not want_los,
            )
            if bool(placement.is_los) == want_los:
                if target_distance_m is not None:
                    error = abs(float(placement.distance_m) - float(target_distance_m))
                    if best_near_distance is None or error < best_near_distance[0]:
                        best_near_distance = (error, scene, placement, attempts)
                    if error > distance_tolerance_m:
                        continue
                return scene, placement, attempts
    if best_near_distance is not None:
        _, scene, placement, found_attempts = best_near_distance
        return scene, placement, found_attempts
    raise RuntimeError(f"Could not sample {'LOS' if want_los else 'NLOS'} placement for {scene_type}")


def _plot_poly_2d(
    ax: Any,
    geom: Any,
    facecolor: str,
    edgecolor: str,
    alpha: float = 1.0,
    linestyle: str = "-",
    linewidth: float = 1.1,
    hatch: str | None = None,
) -> None:
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            hatch=hatch,
            zorder=1,
        )
        for interior in poly.interiors:
            hx, hy = interior.xy
            ax.fill(hx, hy, facecolor="white", edgecolor=edgecolor, linewidth=0.8, zorder=2)


def plot_scene_2d(scene: Scene2D, placement: AcousticPlacement, output_path: Path) -> None:
    if plt is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=180)
    is_open = scene.scene_type in {"open_field", "obstacle_forest"}
    boundary_edge = "#6f6f6f" if is_open else "#222222"
    boundary_style = (0, (4, 3)) if is_open else "-"
    _plot_poly_2d(
        ax,
        scene.boundary,
        facecolor="#fafafa" if is_open else "#f3f3f3",
        edgecolor=boundary_edge,
        alpha=1.0,
        linestyle=boundary_style,
        linewidth=1.2,
    )
    for obstacle in scene.obstacles:
        _plot_poly_2d(
            ax,
            obstacle,
            facecolor="#b8b8b8",
            edgecolor="#333333",
            alpha=1.0,
            linewidth=0.9,
            hatch="///" if scene.scene_type == "obstacle_forest" else None,
        )

    rx, ry, _ = placement.receiver_xyz
    sx, sy, _ = placement.source_xyz
    path_color = "#c62828" if not placement.is_los else "#2f6fb0"
    ax.plot([sx, rx], [sy, ry], linestyle=(0, (4, 3)), color=path_color, linewidth=1.4, zorder=4)
    ax.scatter([sx], [sy], s=140, marker="*", color="#d62728", edgecolor="white", linewidth=0.9, zorder=5)
    ax.scatter([rx], [ry], s=66, marker="o", color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=5)

    legend_handles = [
        Line2D([0], [0], color=boundary_edge, linestyle=boundary_style, linewidth=1.2, label="开放边界" if is_open else "场景边界"),
        Line2D([0], [0], color=path_color, linestyle=(0, (4, 3)), linewidth=1.4, label="声源-接收器连线"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728", markeredgecolor="white", markersize=12, label="声源"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="white", markersize=7, label="接收器"),
    ]
    if scene.obstacles:
        legend_handles.append(Patch(facecolor="#b8b8b8", edgecolor="#333333", label="遮挡物"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, prop=FONT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _poly_to_faces(poly: Any, z0: float, z1: float) -> list[list[tuple[float, float, float]]]:
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    bottom = [(float(x), float(y), float(z0)) for x, y in coords]
    top = [(float(x), float(y), float(z1)) for x, y in coords]
    faces: list[list[tuple[float, float, float]]] = [bottom, top]
    for idx in range(len(coords)):
        nxt = (idx + 1) % len(coords)
        faces.append([bottom[idx], bottom[nxt], top[nxt], top[idx]])
    return faces


def _add_poly3d(ax: Any, geom: Any, z0: float, z1: float, color: str, edge: str, alpha: float) -> None:
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in geoms:
        faces = _poly_to_faces(poly, z0, z1)
        collection = Poly3DCollection(faces, facecolors=color, edgecolors=edge, linewidths=0.55, alpha=alpha)
        ax.add_collection3d(collection)


def _add_floor3d(ax: Any, geom: Any, color: str, edge: str, alpha: float) -> None:
    geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in geoms:
        coords = list(poly.exterior.coords)
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        face = [[(float(x), float(y), 0.0) for x, y in coords]]
        collection = Poly3DCollection(face, facecolors=color, edgecolors=edge, linewidths=0.7, alpha=alpha)
        ax.add_collection3d(collection)


def _draw_open_boundary_wireframe(ax: Any, scene: Scene2D) -> None:
    coords = list(scene.boundary.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    bottom = [(float(x), float(y), 0.0) for x, y in coords]
    top = [(float(x), float(y), float(scene.height_m)) for x, y in coords]
    for ring, linestyle in [(bottom, "-"), (top, (0, (4, 3)))]:
        closed = ring + [ring[0]]
        ax.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            [p[2] for p in closed],
            color="#7d7d7d",
            linestyle=linestyle,
            linewidth=0.9,
        )
    for lower, upper in zip(bottom, top):
        ax.plot(
            [lower[0], upper[0]],
            [lower[1], upper[1]],
            [lower[2], upper[2]],
            color="#8a8a8a",
            linestyle=(0, (3, 4)),
            linewidth=0.75,
            alpha=0.65,
        )


def plot_scene_3d(scene: Scene2D, placement: AcousticPlacement, output_path: Path) -> None:
    if plt is None or Poly3DCollection is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    if scene.is_outdoor:
        _add_floor3d(ax, scene.boundary, "#d8ead0", "#6f6f6f", 0.28)
        _draw_open_boundary_wireframe(ax, scene)
    else:
        _add_poly3d(ax, scene.boundary, 0.0, scene.height_m, "#e7e7e3", "#303030", 0.18)
    for obstacle in scene.obstacles:
        obstacle_height = scene.height_m if not scene.is_outdoor else min(scene.height_m, 3.5)
        _add_poly3d(ax, obstacle, 0.0, obstacle_height, "#777777", "#202020", 0.78)

    sx, sy, sz = placement.source_xyz
    rx, ry, rz = placement.receiver_xyz
    path_color = "#c62828" if not placement.is_los else "#2f6fb0"
    ax.plot([sx, rx], [sy, ry], [sz, rz], color=path_color, linestyle=(0, (4, 3)), linewidth=1.7)
    ax.scatter([sx], [sy], [sz], color="#d62728", marker="*", s=90, depthshade=False)
    ax.scatter([rx], [ry], [rz], color="#1f77b4", marker="o", s=64, depthshade=False)

    minx, miny, maxx, maxy = scene.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_zlim(0.0, scene.height_m)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    legend_handles = [
        Line2D([0], [0], color=path_color, linestyle=(0, (4, 3)), linewidth=1.7, label="声源-接收器连线"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728", markersize=9, label="声源"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=7, label="接收器"),
    ]
    if scene.obstacles:
        legend_handles.append(Patch(facecolor="#777777", edgecolor="#202020", label="遮挡物"))
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#7d7d7d" if scene.is_outdoor else "#303030",
            linestyle=(0, (4, 3)) if scene.is_outdoor else "-",
            linewidth=1.0,
            label="开放边界" if scene.is_outdoor else "房间边界",
        )
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, prop=FONT)
    ax.view_init(elev=28, azim=-52)
    try:
        ax.set_box_aspect((maxx - minx, maxy - miny, scene.height_m))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def convolve_probe_mono(probe: np.ndarray, mono_rir: np.ndarray, sample_count: int) -> np.ndarray:
    rendered = np.convolve(np.asarray(probe, dtype=np.float64), np.asarray(mono_rir, dtype=np.float64), mode="full")
    return rendered[:sample_count].astype(np.float32)


def normalize_rir_for_plot(rir: np.ndarray) -> np.ndarray:
    arr = np.asarray(rir, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[1]:
        return arr.T
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D RIR, got shape {arr.shape}")
    return arr


def schroeder_decay_db(signal: np.ndarray) -> np.ndarray:
    energy = np.asarray(signal, dtype=np.float64) ** 2
    if energy.size == 0:
        return np.zeros(0, dtype=np.float64)
    decay = np.cumsum(energy[::-1])[::-1]
    peak = max(float(decay[0]), 1e-20)
    return 10.0 * np.log10(np.maximum(decay / peak, 1e-20))


def estimate_reverb_time(decay_db: np.ndarray, sample_rate: int, low_db: float, high_db: float) -> float:
    if decay_db.size < 2:
        return float("nan")
    indices = np.flatnonzero((decay_db <= low_db) & (decay_db >= high_db))
    if indices.size < max(8, int(round(0.005 * sample_rate))):
        return float("nan")
    t = indices.astype(np.float64) / float(sample_rate)
    y = decay_db[indices].astype(np.float64)
    slope, _ = np.polyfit(t, y, 1)
    if slope >= -1e-9:
        return float("nan")
    return float(-60.0 / slope)


def plot_rir_diagnostics(
    rir: np.ndarray,
    sample_rate: int,
    output_path: Path,
    expected_sample: int | None,
    observed_sample: int | None,
    window_ms: float,
) -> None:
    if plt is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = normalize_rir_for_plot(rir)
    mono = arr[:, 0] * math.sqrt(2.0)
    t_ms = np.arange(arr.shape[0], dtype=np.float64) / float(sample_rate) * 1000.0
    mono_abs = np.abs(mono)
    mono_peak = max(float(np.max(mono_abs)) if mono_abs.size else 0.0, 1e-12)
    mono_db = 20.0 * np.log10(np.maximum(mono_abs / mono_peak, 1e-12))
    decay_db = schroeder_decay_db(mono)
    rt20 = estimate_reverb_time(decay_db, sample_rate, -5.0, -25.0)
    rt30 = estimate_reverb_time(decay_db, sample_rate, -5.0, -35.0)
    rt60 = estimate_reverb_time(decay_db, sample_rate, -5.0, -65.0)

    x_max = float(t_ms[-1]) if t_ms.size else 0.0
    if window_ms > 0.0:
        marker_samples = [sample for sample in (expected_sample, observed_sample) if sample is not None]
        marker_ms = [sample / float(sample_rate) * 1000.0 for sample in marker_samples]
        x_max = min(x_max, max(float(window_ms), *(value + 8.0 for value in marker_ms)) if marker_ms else float(window_ms))

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.0), dpi=150, sharex=True)
    axes[0].plot(t_ms, mono, color="#2f2f2f", linewidth=0.8, zorder=3)
    axes[0].set_ylabel("单通道RIR幅值", **_font_kwargs())
    if mono_abs.size:
        axes[0].set_ylim(-mono_peak * 1.12, mono_peak * 1.12)
    axes[0].grid(True, linewidth=0.3, alpha=0.3)

    axes[1].plot(t_ms, mono_db, color="#1f77b4", linewidth=1.0, zorder=3)
    axes[1].set_ylabel("单通道幅值 / dB", **_font_kwargs())
    axes[1].set_ylim(-80.0, 3.0)
    axes[1].grid(True, linewidth=0.3, alpha=0.3)

    axes[2].plot(t_ms[: decay_db.size], decay_db, color="#55a868", linewidth=1.2, zorder=3)
    axes[2].axhline(-5.0, color="#777777", linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.7)
    axes[2].axhline(-25.0, color="#777777", linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.7)
    axes[2].axhline(-35.0, color="#999999", linewidth=0.6, linestyle=(0, (2, 4)), alpha=0.7)
    axes[2].axhline(-65.0, color="#bbbbbb", linewidth=0.6, linestyle=(0, (1, 4)), alpha=0.7)
    axes[2].set_xlabel("时间 / ms", **_font_kwargs())
    axes[2].set_ylabel("Schroeder能量 / dB", **_font_kwargs())
    axes[2].set_ylim(-80.0, 3.0)
    axes[2].grid(True, linewidth=0.3, alpha=0.3)
    if x_max > 0.0:
        axes[2].set_xlim(0.0, x_max)
    for ax in axes:
        transform = ax.get_xaxis_transform()
        if expected_sample is not None:
            expected_ms = expected_sample / float(sample_rate) * 1000.0
            ax.scatter([expected_ms], [0.98], marker="v", s=28, color="#d62728", transform=transform, clip_on=False, zorder=5)
        if observed_sample is not None:
            observed_ms = observed_sample / float(sample_rate) * 1000.0
            ax.scatter([observed_ms], [0.90], marker="v", s=28, color="#2ca02c", transform=transform, clip_on=False, zorder=5)

    handles: list[Any] = []
    if expected_sample is not None:
        handles.append(Line2D([0], [0], color="#d62728", marker="v", linestyle="None", markersize=5, label="几何直达延迟"))
    if observed_sample is not None:
        handles.append(Line2D([0], [0], color="#2ca02c", marker="v", linestyle="None", markersize=5, label="观测首到达"))
    handles.append(Line2D([0], [0], color="#1f77b4", linewidth=1.0, label="单通道RIR幅值"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, frameon=True, prop=FONT)
    rt_label_parts = []
    if np.isfinite(rt20):
        rt_label_parts.append(f"RT20={rt20:.2f}s")
    if np.isfinite(rt30):
        rt_label_parts.append(f"RT30={rt30:.2f}s")
    if np.isfinite(rt60):
        rt_label_parts.append(f"RT60={rt60:.2f}s")
    rt_label = "，".join(rt_label_parts) if rt_label_parts else "RT估计不足"
    axes[2].legend(
        handles=[Line2D([0], [0], color="#55a868", linewidth=1.2, label=rt_label)],
        loc="upper right",
        fontsize=8,
        frameon=True,
        prop=FONT,
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    def decorate_time_axis(ax: Any) -> None:
        if x_max > 0.0:
            ax.set_xlim(0.0, x_max)
        transform = ax.get_xaxis_transform()
        if expected_sample is not None:
            expected_ms = expected_sample / float(sample_rate) * 1000.0
            ax.scatter([expected_ms], [0.98], marker="v", s=28, color="#d62728", transform=transform, clip_on=False, zorder=5)
        if observed_sample is not None:
            observed_ms = observed_sample / float(sample_rate) * 1000.0
            ax.scatter([observed_ms], [0.90], marker="v", s=28, color="#2ca02c", transform=transform, clip_on=False, zorder=5)
        ax.grid(True, linewidth=0.3, alpha=0.3)

    separate_specs = [
        (
            output_path.with_name(f"{output_path.stem}_waveform.png"),
            mono,
            "#2f2f2f",
            "单通道RIR幅值",
            "单通道RIR幅值",
            (-mono_peak * 1.12, mono_peak * 1.12),
            t_ms,
        ),
        (
            output_path.with_name(f"{output_path.stem}_amplitude_db.png"),
            mono_db,
            "#1f77b4",
            "单通道幅值 / dB",
            "单通道RIR幅值",
            (-80.0, 3.0),
            t_ms,
        ),
        (
            output_path.with_name(f"{output_path.stem}_schroeder_decay.png"),
            decay_db,
            "#55a868",
            "Schroeder能量 / dB",
            rt_label,
            (-80.0, 3.0),
            t_ms[: decay_db.size],
        ),
    ]
    for separate_path, values, color, ylabel, legend_label, ylim, t_values in separate_specs:
        subfig, subax = plt.subplots(figsize=(8.4, 3.0), dpi=150)
        subax.plot(t_values, values, color=color, linewidth=1.0)
        if separate_path.name.endswith("_schroeder_decay.png"):
            subax.axhline(-5.0, color="#777777", linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.7)
            subax.axhline(-25.0, color="#777777", linewidth=0.6, linestyle=(0, (3, 3)), alpha=0.7)
            subax.axhline(-35.0, color="#999999", linewidth=0.6, linestyle=(0, (2, 4)), alpha=0.7)
            subax.axhline(-65.0, color="#bbbbbb", linewidth=0.6, linestyle=(0, (1, 4)), alpha=0.7)
        subax.set_xlabel("时间 / ms", **_font_kwargs())
        subax.set_ylabel(ylabel, **_font_kwargs())
        subax.set_ylim(*ylim)
        decorate_time_axis(subax)
        sub_handles = [Line2D([0], [0], color=color, linewidth=1.0, label=legend_label)]
        if expected_sample is not None:
            sub_handles.append(
                Line2D([0], [0], color="#d62728", marker="v", linestyle="None", markersize=5, label="几何直达延迟")
            )
        if observed_sample is not None:
            sub_handles.append(
                Line2D([0], [0], color="#2ca02c", marker="v", linestyle="None", markersize=5, label="观测首到达")
            )
        subax.legend(
            handles=sub_handles,
            loc="upper right",
            fontsize=8,
            frameon=True,
            prop=FONT,
        )
        subfig.tight_layout()
        subfig.savefig(separate_path)
        plt.close(subfig)


def plot_rir_overlay(records: list[dict[str, Any]], sample_rate: int, output_path: Path, window_ms: float) -> None:
    if plt is None or not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    colors = ["#c44e52", "#4c72b0", "#dd8452", "#55a868", "#8172b3", "#64b5cd"]
    mono_abs_values: list[np.ndarray] = []
    for record in records:
        arr = normalize_rir_for_plot(record["rir"])
        mono_abs_values.append(np.abs(arr[:, 0] * math.sqrt(2.0)))
    global_peak = max([float(np.max(value)) for value in mono_abs_values if value.size] or [1e-12])
    x_max = 0.0
    for index, record in enumerate(records):
        mono_abs = mono_abs_values[index]
        mono_db = 20.0 * np.log10(np.maximum(mono_abs / max(global_peak, 1e-12), 1e-12))
        t_ms = np.arange(mono_abs.shape[0], dtype=np.float64) / float(sample_rate) * 1000.0
        if t_ms.size:
            x_max = max(x_max, float(t_ms[-1]))
        label = f"{record['label']} ({record['actual']})"
        ax.plot(t_ms, mono_db, linewidth=1.0, color=colors[index % len(colors)], label=label)
        observed = record.get("observed_sample")
        if observed is not None:
            observed_ms = observed / float(sample_rate) * 1000.0
            x_max = max(x_max, observed_ms + 8.0)
            sample = min(int(observed), mono_db.size - 1)
            ax.scatter(
                [observed_ms],
                [mono_db[sample] if sample >= 0 else 0.0],
                marker="o",
                s=18,
                color=colors[index % len(colors)],
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
    ax.set_xlabel("时间 / ms", **_font_kwargs())
    ax.set_ylabel("单通道RIR幅值 / dB（全局归一）", **_font_kwargs())
    ax.set_ylim(-80.0, 3.0)
    if window_ms > 0.0:
        ax.set_xlim(0.0, min(x_max, max(float(window_ms), 1.0)))
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, frameon=True, prop=FONT)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_schroeder_overlay(records: list[dict[str, Any]], sample_rate: int, output_path: Path, window_ms: float) -> None:
    if plt is None or not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    colors = ["#c44e52", "#4c72b0", "#dd8452", "#55a868", "#8172b3", "#64b5cd"]
    x_max = 0.0
    for index, record in enumerate(records):
        arr = normalize_rir_for_plot(record["rir"])
        mono = arr[:, 0] * math.sqrt(2.0)
        decay_db = schroeder_decay_db(mono)
        t_ms = np.arange(decay_db.shape[0], dtype=np.float64) / float(sample_rate) * 1000.0
        if t_ms.size:
            x_max = max(x_max, float(t_ms[-1]))
        label = f"{record['label']} ({record['actual']})"
        ax.plot(t_ms, decay_db, linewidth=1.1, color=colors[index % len(colors)], label=label)
    ax.set_xlabel("时间 / ms", **_font_kwargs())
    ax.set_ylabel("Schroeder能量 / dB", **_font_kwargs())
    ax.set_ylim(-80.0, 3.0)
    if window_ms > 0.0:
        ax.set_xlim(0.0, min(x_max, max(float(window_ms), 1.0)))
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, frameon=True, prop=FONT)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def rir_energy_metrics_for_manifest(
    rir: np.ndarray,
    sample_rate: int,
    direct_sample: int | None = None,
) -> dict[str, float]:
    arr = normalize_rir_for_plot(rir)
    envelope = np.sqrt(np.sum(arr * arr, axis=1))
    mono = arr[:, 0] * math.sqrt(2.0)
    decay_db = schroeder_decay_db(mono)
    rt20 = estimate_reverb_time(decay_db, sample_rate, -5.0, -25.0)
    rt30 = estimate_reverb_time(decay_db, sample_rate, -5.0, -35.0)
    rt60 = estimate_reverb_time(decay_db, sample_rate, -5.0, -65.0)
    energy = envelope * envelope
    total = float(np.sum(energy))
    direct_energy = 0.0
    if direct_sample is not None:
        half_window = max(1, int(round(0.002 * sample_rate)))
        start = max(0, int(direct_sample) - half_window)
        stop = min(energy.size, int(direct_sample) + half_window + 1)
        direct_energy = float(np.sum(energy[start:stop]))
    if total <= 0.0:
        return {
            "rir_total_energy": 0.0,
            "rir_direct_window_energy": 0.0,
            "rir_late20_energy_ratio": 0.0,
            "rir_late50_energy_ratio": 0.0,
            "rir_late100_energy_ratio": 0.0,
            "rir_late20_to_direct_db": float("nan"),
            "rir_late50_to_direct_db": float("nan"),
            "rir_rt20_s": float("nan"),
            "rir_rt30_s": float("nan"),
            "rir_rt60_s": float("nan"),
        }
    late20 = float(np.sum(energy[int(round(0.020 * sample_rate)) :]))
    late50 = float(np.sum(energy[int(round(0.050 * sample_rate)) :]))
    late100 = float(np.sum(energy[int(round(0.100 * sample_rate)) :]))
    return {
        "rir_total_energy": total,
        "rir_direct_window_energy": direct_energy,
        "rir_late20_energy_ratio": late20 / total,
        "rir_late50_energy_ratio": late50 / total,
        "rir_late100_energy_ratio": late100 / total,
        "rir_late20_to_direct_db": 10.0 * math.log10(max(late20, 1e-20) / max(direct_energy, 1e-20)),
        "rir_late50_to_direct_db": 10.0 * math.log10(max(late50, 1e-20) / max(direct_energy, 1e-20)),
        "rir_rt20_s": rt20,
        "rir_rt30_s": rt30,
        "rir_rt60_s": rt60,
    }


def plot_late_energy_bars(rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None or not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [SCENE_TITLES.get(str(row["scene_type"]), str(row["scene_type"])) for row in rows]
    late20 = [float(row["rir_late20_energy_ratio"]) for row in rows]
    late20_direct_db = [float(row["rir_late20_to_direct_db"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), dpi=150, sharex=True)
    axes[0].bar(x, late20, width=0.55, color="#4c72b0", label="20ms后能量占比")
    axes[0].set_ylabel("能量占比", **_font_kwargs())
    axes[0].set_ylim(0.0, max(0.65, max(late20) * 1.15))
    axes[0].grid(True, axis="y", linewidth=0.3, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, frameon=True, prop=FONT)

    axes[1].bar(x, late20_direct_db, width=0.55, color="#55a868", label="20ms后/直达窗")
    axes[1].axhline(0.0, color="#333333", linewidth=0.7, alpha=0.6)
    axes[1].set_ylabel("相对直达 / dB", **_font_kwargs())
    finite = [value for value in late20_direct_db if np.isfinite(value)]
    if finite:
        axes[1].set_ylim(min(-80.0, min(finite) - 3.0), max(6.0, max(finite) + 3.0))
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, frameon=True, prop=FONT)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", **_font_kwargs())
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_material_damping_scale(float(args.material_damping_scale))
    set_material_medium(args.material_medium_density, args.material_medium_speed)
    if args.indoor_surfaces_gypsum:
        apply_indoor_surfaces_gypsum_override()
    if args.indoor_balanced_reflective:
        apply_indoor_balanced_reflective_override()

    availability = check_soundspaces_available()
    if not availability.get("available"):
        print(json.dumps({"passed": False, "reason": "soundspaces_unavailable", "detail": availability}, indent=2))
        return 2

    out = args.output_dir.resolve()
    geometry_dir = out / "geometry"
    audio_dir = out / "audio"
    dry_dir = out / "dry"
    figure_dir = out / "figures"
    report_dir = out / "reports"
    cases_dir = out / "cases"
    rirs_dir = out / "rirs"
    for path in (geometry_dir, audio_dir, dry_dir, figure_dir, report_dir, cases_dir, rirs_dir):
        path.mkdir(parents=True, exist_ok=True)

    material_db_path = report_dir / "occ_rlr_materials.json"
    material_db = write_occ_material_database(material_db_path)
    material_assignment_path = report_dir / "occ_scene_material_assignments.json"
    material_assignments = write_scene_material_assignments(material_assignment_path)
    config = SoundSpacesConfig(
        sample_rate=args.sample_rate,
        ir_duration_s=args.ir_duration,
        direct_ray_count=args.direct_ray_count,
        indirect_ray_count=args.indirect_ray_count,
        indirect_ray_depth=args.indirect_ray_depth,
        source_ray_count=args.source_ray_count,
        source_ray_depth=args.source_ray_depth,
        thread_count=args.thread_count,
        output_directory=str(out),
        align_output_onset=bool(args.align_output_onset),
        onset_threshold_db=args.onset_threshold_db,
        enable_materials=not args.disable_materials,
        audio_materials_json=str(material_db_path) if not args.disable_materials else None,
        transmission=not bool(args.disable_transmission),
        enable_rgb=False,
        enable_depth=False,
    )
    config.save_json(report_dir / "soundspaces_config.json")
    outdoor_boundary_mode = args.outdoor_boundary_mode
    if outdoor_boundary_mode == "auto":
        outdoor_boundary_mode = "geometric_open" if args.disable_materials else "absorbing_shell"

    dry = make_short_impulse_probe(args.sample_rate, args.duration, args.impulse_time, args.impulse_duration)
    dry_wav = dry_dir / "short_impulse_probe.wav"
    sf.write(dry_wav, dry, args.sample_rate)
    np.save(dry_dir / "short_impulse_probe.npy", dry)
    if not args.no_plots:
        save_spectrum_plot(dry, args.sample_rate, figure_dir / "short_impulse_probe_spectrum.png", "短冲激探针频谱")
        save_stft_plot(
            dry,
            args.sample_rate,
            figure_dir / "short_impulse_probe_stft.png",
            "短冲激探针 STFT",
            args.stft_n_fft,
            args.stft_hop_length,
        )
    dry_metrics = spectrum_summary(dry, args.sample_rate)

    rng = random.Random(args.seed)
    scenes = generate_all_scenes(variants_per_type=args.variants_per_type, seed=args.seed)
    selected = []
    for scene_type in SCENE_ORDER:
        target_distance = None
        if TARGET_LOS[scene_type] and args.los_control_distance > 0.0:
            target_distance = float(args.los_control_distance)
        selected.append(
            select_scene_and_placement(
                scene_type,
                scenes,
                rng,
                target_distance_m=target_distance,
                distance_tolerance_m=float(args.distance_tolerance),
            )
        )

    rows: list[dict[str, Any]] = []
    rir_plot_records: list[dict[str, Any]] = []
    for scene_index, (scene, placement, attempts) in enumerate(selected):
        files = export_scene_obj(scene, geometry_dir, outdoor_boundary_mode=outdoor_boundary_mode)
        case_id = f"probe_{scene_index:03d}_{scene.scene_id}"
        case_dir = cases_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_config = replace(config, direct=bool(placement.is_los) or not bool(args.disable_direct_for_nlos))
        case_config.save_json(case_dir / "soundspaces_case_config.json")
        backend = SoundSpacesBackend(case_config)
        rir = backend.render_rir(
            scene_mesh_path=Path(files["obj"]),
            source_occ_xyz=placement.source_xyz,
            receiver_occ_xyz=placement.receiver_xyz,
            output_dir=case_dir,
        )
        rir_path = rirs_dir / f"{case_id}_rir.npy"
        np.save(rir_path, np.asarray(rir, dtype=np.float32))
        validation = validate_rir_physics(rir, placement.source_xyz, placement.receiver_xyz, args.sample_rate, placement.is_los)

        foa_path = audio_dir / f"{case_id}_foa.wav"
        mono_path = audio_dir / f"{case_id}_mono.wav"
        audio_meta = backend.convolve_and_save(dry, rir, foa_path, mono_path)
        mono, sr = sf.read(mono_path, always_2d=False)
        if int(sr) != int(args.sample_rate):
            raise RuntimeError(f"unexpected sample rate in {mono_path}: {sr}")
        mono = np.asarray(mono, dtype=np.float32)
        mono_metrics = spectrum_summary(mono, args.sample_rate)
        direct_sample_for_metrics = (
            validation.expected_direct_delay_sample if bool(placement.is_los) else validation.observed_first_peak_sample
        )
        rir_energy_metrics = rir_energy_metrics_for_manifest(rir, args.sample_rate, direct_sample_for_metrics)
        mono_rir = mono_rir_from_foa(rir)
        unaligned_impulse = convolve_probe_mono(dry, mono_rir, dry.shape[0])
        unaligned_path = audio_dir / f"{case_id}_mono_unaligned_impulse.wav"
        sf.write(unaligned_path, unaligned_impulse, args.sample_rate)

        if not args.no_plots:
            plot_scene_2d(scene, placement, figure_dir / f"{case_id}_layout_2d.png")
            plot_scene_3d(scene, placement, figure_dir / f"{case_id}_layout_3d.png")
            scene_title = SCENE_TITLES[scene.scene_type]
            plot_rir_diagnostics(
                rir,
                args.sample_rate,
                figure_dir / f"{case_id}_rir_summary.png",
                validation.expected_direct_delay_sample,
                validation.observed_first_peak_sample,
                args.rir_plot_window_ms,
            )
            save_spectrum_plot(mono, args.sample_rate, figure_dir / f"{case_id}_receiver_spectrum.png", f"{scene_title}接收信号频谱")
            save_stft_plot(
                mono,
                args.sample_rate,
                figure_dir / f"{case_id}_receiver_stft.png",
                f"{scene_title}接收信号 STFT",
                args.stft_n_fft,
                args.stft_hop_length,
            )
            save_delta_stft_plot(
                dry,
                mono,
                args.sample_rate,
                figure_dir / f"{case_id}_receiver_minus_source_stft.png",
                f"{scene_title}接收信号 - 发射信号 STFT",
                args.stft_n_fft,
                args.stft_hop_length,
            )
            save_transfer_plot(
                dry,
                mono,
                args.sample_rate,
                figure_dir / f"{case_id}_receiver_source_transfer.png",
                f"{scene_title}接收/发射幅度传递",
            )

        row = {
            "case_id": case_id,
            "scene_id": scene.scene_id,
            "scene_type": scene.scene_type,
            "variant_index": scene.variant_index,
            "target": "LOS" if TARGET_LOS[scene.scene_type] else "NLOS",
            "actual": "LOS" if placement.is_los else "NLOS",
            "attempts_until_match": attempts,
            "is_outdoor": bool(scene.is_outdoor),
            "floor_material": scene_material_assignment(scene.scene_type)["floor"] or "",
            "wall_material": scene_material_assignment(scene.scene_type)["wall"] or "",
            "ceiling_material": scene_material_assignment(scene.scene_type)["ceiling"] or "",
            "obstacle_material": scene_material_assignment(scene.scene_type)["obstacle"] or "",
            "open_boundary_material": scene_material_assignment(scene.scene_type)["open_boundary"] or "",
            "open_ceiling_material": scene_material_assignment(scene.scene_type)["open_ceiling"] or "",
            "prefer_obstructed": not TARGET_LOS[scene.scene_type],
            "is_los": bool(placement.is_los),
            "rir_direct_enabled": bool(case_config.direct),
            "rir_transmission_enabled": bool(case_config.transmission),
            "obstruction_count": int(placement.obstruction_count),
            "obstruction_types": "|".join(placement.obstruction_types),
            "distance_m": float(placement.distance_m),
            "target_distance_m": (
                float(args.los_control_distance)
                if TARGET_LOS[scene.scene_type] and args.los_control_distance > 0.0
                else ""
            ),
            "target_distance_error_m": (
                float(placement.distance_m - args.los_control_distance)
                if TARGET_LOS[scene.scene_type] and args.los_control_distance > 0.0
                else ""
            ),
            "azimuth_rad": float(placement.azimuth_rad),
            "elevation_rad": float(placement.elevation_rad),
            "source_x": float(placement.source_xyz[0]),
            "source_y": float(placement.source_xyz[1]),
            "source_z": float(placement.source_xyz[2]),
            "receiver_x": float(placement.receiver_xyz[0]),
            "receiver_y": float(placement.receiver_xyz[1]),
            "receiver_z": float(placement.receiver_xyz[2]),
            "obj_path": str(Path(files["obj"]).relative_to(out)),
            "rir_path": str(rir_path.relative_to(out)),
            "foa_wav_path": str(foa_path.relative_to(out)),
            "mono_wav_path": str(mono_path.relative_to(out)),
            "mono_unaligned_impulse_wav_path": str(unaligned_path.relative_to(out)),
            "rir_shape": json.dumps(list(np.asarray(rir).shape)),
            **rir_energy_metrics,
            "validation_passed": bool(validation.passed),
            "validation_expected_direct_delay_sample": int(validation.expected_direct_delay_sample),
            "validation_observed_first_peak_sample": validation.observed_first_peak_sample,
            "dry_rms": dry_metrics["rms"],
            "dry_peak": dry_metrics["peak"],
            "dry_spectral_centroid_hz": dry_metrics["spectral_centroid_hz"],
            "dry_hf_ratio_2k_nyquist": dry_metrics["hf_ratio_2k_nyquist"],
            "mono_rms": mono_metrics["rms"],
            "mono_peak": mono_metrics["peak"],
            "mono_spectral_centroid_hz": mono_metrics["spectral_centroid_hz"],
            "mono_hf_ratio_2k_nyquist": mono_metrics["hf_ratio_2k_nyquist"],
            "alignment_onset_sample": (audio_meta.get("alignment") or {}).get("onset_sample", ""),
        }
        rows.append(row)
        rir_plot_records.append(
            {
                "label": SCENE_TITLES[scene.scene_type],
                "actual": row["actual"],
                "rir": np.asarray(rir, dtype=np.float64),
                "observed_sample": validation.observed_first_peak_sample,
            }
        )
        print(json.dumps({"rendered": len(rows), "case_id": case_id, "actual": row["actual"]}, ensure_ascii=False))

    if not args.no_plots:
        plot_rir_overlay(
            rir_plot_records,
            args.sample_rate,
            figure_dir / "six_scene_rir_envelope_overlay.png",
            args.rir_plot_window_ms,
        )
        plot_schroeder_overlay(
            rir_plot_records,
            args.sample_rate,
            figure_dir / "six_scene_schroeder_decay_overlay.png",
            args.rir_plot_window_ms,
        )
        plot_late_energy_bars(rows, figure_dir / "six_scene_late_energy_ratio.png")

    write_manifest(out / "probe_manifest.csv", rows)
    write_json(out / "probe_manifest.json", rows)
    write_json(
        report_dir / "run_summary.json",
        {
            "output_dir": str(out),
            "scene_count": len(rows),
            "scene_order": SCENE_ORDER,
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "impulse_time": args.impulse_time,
            "impulse_duration": args.impulse_duration,
            "los_control_distance": args.los_control_distance,
            "distance_tolerance": args.distance_tolerance,
            "dry_wav": str(dry_wav),
            "material_database": str(material_db_path),
            "material_database_payload": material_db,
            "material_assignments": str(material_assignment_path),
            "material_assignments_payload": material_assignments,
            "allow_transmission": not bool(args.disable_transmission),
            "disable_transmission": bool(args.disable_transmission),
            "disable_materials": bool(args.disable_materials),
            "indoor_surfaces_gypsum": bool(args.indoor_surfaces_gypsum),
            "indoor_balanced_reflective": bool(args.indoor_balanced_reflective),
            "material_damping_scale": float(args.material_damping_scale),
            "material_medium_density": args.material_medium_density,
            "material_medium_speed": args.material_medium_speed,
            "keep_direct_for_nlos": not bool(args.disable_direct_for_nlos),
            "disable_direct_for_nlos": bool(args.disable_direct_for_nlos),
            "align_output_onset": bool(args.align_output_onset),
            "rir_plot_window_ms": args.rir_plot_window_ms,
            "outdoor_boundary_mode": outdoor_boundary_mode,
            "soundspaces_config": config.to_dict(),
            "dry_metrics": dry_metrics,
            "soundspaces": availability,
            "manifest_csv": str(out / "probe_manifest.csv"),
            "notes": [
                "One random scene is selected from each of the six OCC structure types.",
                "Baffle room, L-shaped corridor, T-shaped corridor, and obstacle forest are forced to NLOS placements.",
                "Empty room and open field are LOS controls because they cannot construct geometric occlusion.",
                "Materials are enabled by default and use reports/occ_rlr_materials.json; pass --disable-materials for the legacy default-material path.",
                "Outdoor boundary mode is auto-selected so material-disabled legacy runs use true geometric open boundaries.",
                "Acoustic transmission is enabled by default; pass --disable-transmission only for ablation examples.",
                "The direct acoustic component is enabled by default for all cases; pass --disable-direct-for-nlos only for schematic occlusion-only examples.",
                "The dry probe is ten seconds long and contains one single-sample delta impulse by default.",
                "RIR summary figures plot the single-channel RIR used for analysis/convolution and mark geometric/observed delays.",
                "RIR summary figures include Schroeder energy decay and RT20/RT30/RT60 estimates when enough decay range is available.",
                "LOS control scenes are sampled toward the same source-receiver distance for fairer indoor/outdoor comparison.",
                "The late-energy comparison figure reports both absolute late-energy ratio and late/direct energy in dB.",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
