#!/usr/bin/env python3
"""Generate architecture diagrams for Evo presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "docs" / "deck_images"

BG = "#0c1220"
PANEL = "#151d2e"
ACCENT = "#38bdf8"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#34d399"
ORANGE = "#fb923c"


def _style_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")


def _box(ax, x, y, w, h, text, color=PANEL, edge=ACCENT, fs=9):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=TEXT, fontsize=fs, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    _style_ax(ax)
    ax.set_title("Deployment architecture", color=TEXT, fontsize=14, pad=12, loc="left", weight="bold")
    _box(ax, 0.4, 4.2, 2.2, 0.9, "Browser\n(evac-evo.vercel.app)")
    _box(ax, 3.2, 4.2, 2.4, 0.9, "Vercel CDN\nstatic Vite build\n/api/* rewrites")
    _box(ax, 6.4, 4.2, 3.0, 0.9, "Oracle ARM VM :8092\nFastAPI + Evo inference")
    _arrow(ax, 2.6, 4.65, 3.2, 4.65)
    _arrow(ax, 5.6, 4.65, 6.4, 4.65)
    feeds = ["NOAA/NWS", "USGS EEW", "GDACS", "NASA FIRMS", "FEMA IPAWS", "PeopleSense GET"]
    for i, f in enumerate(feeds):
        _box(ax, 0.5 + (i % 3) * 3.1, 2.5 - (i // 3) * 1.1, 2.6, 0.7, f, edge=MUTED, fs=8)
        _arrow(ax, 1.8 + (i % 3) * 3.1, 2.5 - (i // 3) * 1.1 + 0.7, 7.9, 4.2)
    _box(ax, 6.6, 0.5, 2.6, 0.8, "SQLite / Neon\nDisaster history", edge=GREEN, fs=8)
    _arrow(ax, 7.9, 4.2, 7.9, 1.3)
    _box(ax, 0.4, 0.4, 2.8, 0.8, "Local Mac + NCS USB\nlocalhost only", edge=ORANGE, fs=8)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def data_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    _style_ax(ax)
    ax.set_title("Dashboard data pipeline (sync / evo / evo13)", color=TEXT, fontsize=14, pad=12, loc="left", weight="bold")
    steps = [
        (0.3, "GovernmentFeedSync\npoll hazard APIs"),
        (2.5, "AlertProcessor\ncollector + enrich"),
        (4.7, "HazardFeatureBuilder\n+ PeopleSense overlay"),
        (6.9, "EvacuationPredictor\nk-NN / Evo hybrid"),
        (9.1, "Dashboard JSON\n+ disaster history"),
    ]
    for x, label in steps:
        _box(ax, x, 2.2, 1.8, 1.2, label, fs=8)
    for i in range(len(steps) - 1):
        _arrow(ax, steps[i][0] + 1.8, 2.8, steps[i + 1][0], 2.8)
    _box(ax, 2.5, 0.5, 5.5, 0.9, "Feature vector: occupancy, density, hazard severity/magnitude/distance, category, scenario", edge=MUTED, fs=8)
    _arrow(ax, 5.2, 2.2, 5.2, 1.4)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def broadcast_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    _style_ax(ax)
    ax.set_title("Full broadcast pipeline (7 LangChain agents)", color=TEXT, fontsize=14, pad=12, loc="left", weight="bold")
    agents = [
        "Coordinator", "Evac Intel", "Researcher", "Panel", "Writer", "Producer", "Script"
    ]
    for i, name in enumerate(agents):
        _box(ax, 0.3 + i * 1.35, 3.0, 1.15, 0.8, name, fs=7)
        if i:
            _arrow(ax, 0.3 + (i - 1) * 1.35 + 1.15, 3.4, 0.3 + i * 1.35, 3.4)
    _box(ax, 0.3, 1.5, 9.2, 1.0, "Outputs: event log, sitrep, charts, panel commentary, article, production brief, HeyGen broadcast script", fs=8)
    _arrow(ax, 5.0, 3.0, 5.0, 2.5)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def ncs_stack(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    _style_ax(ax)
    ax.set_title("Neural Compute Stick vs OpenVINO", color=TEXT, fontsize=14, pad=12, loc="left", weight="bold")
    _box(ax, 0.5, 3.8, 4.0, 1.0, "Evo ONNX model\nevo1.2.onnx", edge=ACCENT)
    _box(ax, 5.5, 3.8, 4.0, 1.0, "OpenVINO IR\nevo1.2.xml + .bin", edge=ACCENT)
    _box(ax, 1.0, 2.0, 3.0, 1.0, "OpenVINO Runtime\n(CPU or MYRIAD plugin)", edge=GREEN)
    _box(ax, 5.5, 2.0, 3.5, 1.0, "Intel NCS2 USB\nMovidius Myriad X", edge=ORANGE)
    _box(ax, 3.0, 0.4, 4.0, 0.9, "Inference: success/time heads → dashboard", fs=9)
    _arrow(ax, 2.5, 3.8, 2.5, 3.0)
    _arrow(ax, 7.5, 3.8, 7.0, 3.0)
    _arrow(ax, 2.5, 2.0, 4.5, 1.3)
    _arrow(ax, 7.2, 2.0, 5.5, 1.3)
    ax.text(0.5, 5.5, "OpenVINO = driver software · NCS = optional USB accelerator", color=MUTED, fontsize=10)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def evo_hybrid(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    _style_ax(ax)
    ax.set_title("Evo 1.2 hybrid inference policy", color=TEXT, fontsize=14, pad=12, loc="left", weight="bold")
    _box(ax, 0.4, 3.5, 2.0, 0.9, "Live feeds\n+ occupancy")
    _box(ax, 2.8, 3.5, 2.0, 0.9, "Feature encoder\nStandardScaler + one-hot")
    _box(ax, 5.2, 4.0, 1.6, 0.8, "MLP dual-head", fs=8)
    _box(ax, 5.2, 2.8, 1.6, 0.8, "LightGBM", fs=8)
    _box(ax, 7.2, 3.5, 1.6, 0.9, "OOF ensemble", fs=8)
    _box(ax, 9.2, 4.2, 2.2, 0.7, "Time (neural)\nTrain Station only", edge=GREEN, fs=7)
    _box(ax, 9.2, 2.6, 2.2, 0.7, "Success + risk\nk-NN k=25", edge=ORANGE, fs=7)
    for x1, x2 in [(2.4, 2.8), (4.8, 3.7), (4.8, 3.2), (6.8, 3.9), (8.8, 4.5), (8.8, 2.9)]:
        _arrow(ax, x1, 3.9, x2, 3.9)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def generate_all() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in [
        ("diagram-architecture.png", architecture),
        ("diagram-data-pipeline.png", data_pipeline),
        ("diagram-broadcast.png", broadcast_pipeline),
        ("diagram-ncs-stack.png", ncs_stack),
        ("diagram-evo-hybrid.png", evo_hybrid),
    ]:
        p = OUT / name
        fn(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in generate_all():
        print(p)
