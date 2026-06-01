from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Wedge
import numpy as np

from .validation import energy_envelope, expected_delay_sample


def _plot_polygon(ax, polygon, *, facecolor: str, edgecolor: str, alpha: float, linewidth: float = 1.4) -> None:
    x, y = polygon.exterior.xy
    ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.fill(hx, hy, facecolor="white", edgecolor=edgecolor, alpha=1.0, linewidth=linewidth)


def _angle_diff_rad(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _heading_deg_ccw_from_pos_x(vec_xy: tuple[float, float]) -> float:
    return float((np.degrees(np.arctan2(vec_xy[1], vec_xy[0])) + 360.0) % 360.0)


def _draw_camera_view(
    ax,
    receiver_xy: tuple[float, float],
    source_xy: tuple[float, float],
    scene_bounds: tuple[float, float, float, float],
    *,
    hfov_deg: float,
    forward_xy: tuple[float, float] | None,
) -> bool:
    rx, ry = receiver_xy
    sx, sy = source_xy
    if forward_xy is None:
        forward = np.array([sx - rx, sy - ry], dtype=np.float64)
    else:
        forward = np.asarray(forward_xy, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-9:
        return False
    forward /= norm
    heading = float(np.arctan2(forward[1], forward[0]))
    heading_deg = _heading_deg_ccw_from_pos_x((forward[0], forward[1]))
    minx, miny, maxx, maxy = scene_bounds
    diag = float(np.hypot(maxx - minx, maxy - miny))
    source_dist = float(np.hypot(sx - rx, sy - ry))
    radius = max(0.8, min(diag * 0.42, max(source_dist * 1.35, 1.2)))
    half = float(np.deg2rad(hfov_deg) / 2.0)
    theta1 = np.rad2deg(heading - half)
    theta2 = np.rad2deg(heading + half)
    wedge = Wedge(
        (rx, ry),
        radius,
        theta1,
        theta2,
        facecolor="#f4c542",
        edgecolor="#a87800",
        alpha=0.18,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(wedge)
    for sign in (-1.0, 1.0):
        theta = heading + sign * half
        ax.plot(
            [rx, rx + radius * np.cos(theta)],
            [ry, ry + radius * np.sin(theta)],
            color="#a87800",
            linestyle=":",
            linewidth=1.2,
            zorder=3,
        )
    end_x = rx + radius * 0.38 * np.cos(heading)
    end_y = ry + radius * 0.38 * np.sin(heading)
    arrow = FancyArrowPatch(
        (rx, ry),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.8,
        color="#b8860b",
        zorder=6,
    )
    ax.add_patch(arrow)
    ax.text(
        end_x + 0.04 * radius * np.cos(heading),
        end_y + 0.04 * radius * np.sin(heading),
        f" camera {heading_deg:.1f} deg",
        fontsize=8,
        color="#6f5200",
        va="center",
        zorder=8,
    )
    source_angle = float(np.arctan2(sy - ry, sx - rx))
    source_in_view = abs(_angle_diff_rad(source_angle, heading)) <= half and source_dist <= radius
    if source_in_view:
        ax.scatter([sx], [sy], s=145, facecolors="none", edgecolors="#8a1f1a", linewidths=1.5, zorder=8)
    return source_in_view


def plot_geometry_debug(
    scene,
    placement,
    output_path: Path,
    title: str = "SoundSpaces geometry debug",
    *,
    show_camera: bool = False,
    camera_hfov_deg: float = 90.0,
    camera_forward_xy: tuple[float, float] | None = None,
) -> None:
    """Plot source, receiver, and direct-path obstruction in OCC coordinates."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _plot_polygon(ax, scene.boundary, facecolor="#f4f7fb", edgecolor="#5a7da0", alpha=0.55)
    for idx, obstacle in enumerate(scene.obstacles):
        _plot_polygon(ax, obstacle, facecolor="#d9d0c4", edgecolor="#6b5b4b", alpha=0.85)
        pt = obstacle.representative_point()
        ax.text(pt.x, pt.y, f"obs {idx}", ha="center", va="center", fontsize=8, color="#4a4038")
    rx = placement.receiver_xyz
    sx = placement.source_xyz
    source_in_view = False
    if show_camera:
        source_in_view = _draw_camera_view(
            ax,
            (rx[0], rx[1]),
            (sx[0], sx[1]),
            scene.boundary.bounds,
            hfov_deg=camera_hfov_deg,
            forward_xy=camera_forward_xy,
        )
    ax.scatter([rx[0]], [rx[1]], s=70, color="#1f77b4", label="receiver")
    ax.scatter([sx[0]], [sx[1]], s=70, color="#c43c2f", marker="*", label="source")
    linestyle = "-" if placement.is_los else "--"
    color = "#2b2b2b" if placement.is_los else "#b23b2e"
    ax.plot([rx[0], sx[0]], [rx[1], sx[1]], linestyle=linestyle, color=color, linewidth=2.0, label="direct path")
    ax.text(rx[0], rx[1], "  R", va="center", fontsize=10)
    ax.text(sx[0], sx[1], "  S", va="center", fontsize=10)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rir_debug(
    rir: np.ndarray,
    source_xyz: Iterable[float],
    receiver_xyz: Iterable[float],
    sample_rate: int,
    output_path: Path,
    title: str = "RIR physics debug",
) -> None:
    """Plot RIR envelope with theoretical direct-path delay."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(list(source_xyz), dtype=np.float64)
    receiver = np.asarray(list(receiver_xyz), dtype=np.float64)
    distance = float(np.linalg.norm(source - receiver))
    expected = expected_delay_sample(distance, sample_rate)
    envelope = energy_envelope(rir)
    time_ms = np.arange(envelope.shape[0]) / sample_rate * 1000.0
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(time_ms, envelope, color="#2d5f87", linewidth=1.2)
    ax.axvline(expected / sample_rate * 1000.0, color="#c43c2f", linestyle="--", linewidth=1.4)
    ax.text(
        expected / sample_rate * 1000.0,
        float(np.max(envelope)) * 0.92 if envelope.size else 0.0,
        " theoretical direct delay",
        color="#8a2b23",
        fontsize=9,
        va="top",
    )
    ax.set_title(title)
    ax.set_xlabel("time / ms")
    ax.set_ylabel("RIR energy envelope")
    if envelope.size:
        ax.set_xlim(0.0, min(float(time_ms[-1]), 120.0))
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rir_comparison(
    named_rirs: dict[str, np.ndarray],
    sample_rate: int,
    output_path: Path,
    title: str = "RIR comparison",
    xlim_ms: float = 160.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for name, rir in named_rirs.items():
        envelope = energy_envelope(rir)
        if envelope.size == 0:
            continue
        time_ms = np.arange(envelope.shape[0]) / sample_rate * 1000.0
        peak = float(np.max(envelope))
        scaled = envelope / peak if peak > 0.0 else envelope
        ax.plot(time_ms, scaled, linewidth=1.1, label=name)
    ax.set_xlim(0.0, xlim_ms)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.set_xlabel("time / ms")
    ax.set_ylabel("normalized RIR envelope")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
