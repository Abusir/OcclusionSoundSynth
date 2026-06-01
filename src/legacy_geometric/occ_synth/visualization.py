from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .sampling import AcousticPlacement
from .scene_generator import Scene2D


def _plot_poly(ax, geom, facecolor: str, edgecolor: str, alpha: float = 1.0) -> None:
    if geom.geom_type == "Polygon":
        geoms = [geom]
    else:
        geoms = list(geom.geoms)
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=1.2)
        for interior in poly.interiors:
            hx, hy = interior.xy
            ax.fill(hx, hy, facecolor="white", edgecolor=edgecolor, alpha=1.0, linewidth=0.8)

def plot_scene(scene: Scene2D, placement: AcousticPlacement | None, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=140)
    _plot_poly(ax, scene.boundary, "#f4f5f3", "#222222", 0.95)
    for obstacle in scene.obstacles:
        _plot_poly(ax, obstacle, "#6b6f76", "#202020", 1.0)
    if placement is not None:
        rx = placement.receiver_xyz
        sx = placement.source_xyz
        color = "#0a7cff" if placement.is_los else "#d62828"
        ax.plot([rx[0], sx[0]], [rx[1], sx[1]], color=color, linewidth=1.4, linestyle="-" if placement.is_los else "--")
        ax.scatter([rx[0]], [rx[1]], s=55, color="#125fb8", label="FOA receiver")
        ax.scatter([sx[0]], [sx[1]], s=65, color="#d62828", marker="*", label="source")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{scene.scene_id} | {'outdoor' if scene.is_outdoor else 'indoor'}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_catalog_overview(scenes: list[Scene2D], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(6, 10, figsize=(20, 11), dpi=120)
    for ax, scene in zip(axes.flat, scenes):
        _plot_poly(ax, scene.boundary, "#f4f5f3", "#222222", 0.95)
        for obstacle in scene.obstacles:
            _plot_poly(ax, obstacle, "#6b6f76", "#202020", 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{scene.scene_type_id}-{scene.variant_index}", fontsize=7)
    fig.suptitle("60 generated scene variants", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_waveform(wav_path: Path, output_path: Path, max_samples: int = 4000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(wav_path, always_2d=True)
    audio = audio[:max_samples]
    t = np.arange(audio.shape[0]) / sample_rate
    fig, axes = plt.subplots(4, 1, figsize=(8, 5), dpi=140, sharex=True)
    names = ["W", "Y", "Z", "X"]
    for idx, ax in enumerate(axes):
        ax.plot(t, audio[:, idx], linewidth=0.7)
        ax.set_ylabel(names[idx])
        ax.grid(True, linewidth=0.3, alpha=0.35)
    axes[-1].set_xlabel("time / s")
    fig.suptitle(wav_path.name)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
