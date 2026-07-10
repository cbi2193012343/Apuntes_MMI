from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
    }
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figuras"
OUT_DIR.mkdir(exist_ok=True)


def draw_axes(ax: plt.Axes, lim=2.0) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(lim, 0), xytext=(-lim, 0), arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0))
    ax.annotate("", xy=(0, lim), xytext=(0, -lim), arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0))
    ax.text(lim - 0.10, -0.18, r"$x$", fontsize=13)
    ax.text(-0.14, lim - 0.06, r"$y$", fontsize=13)


def spiral_to_cycle(r0: float, stable: bool = True, theta0: float = 0.0, tmax: float = 13.0, n: int = 1200):
    t = np.linspace(0.0, tmax, n)
    k = 0.85 if stable else -0.85
    r = 1.0 + (r0 - 1.0) * np.exp(-k * t)
    theta = theta0 + 1.25 * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def spiral_from_cycle(r0: float, stable: bool = False, theta0: float = 0.0, tmax: float = 10.0, n: int = 1200):
    # If stable=False, the trajectories move away from the cycle.
    t = np.linspace(0.0, tmax, n)
    k = 0.75 if stable else 0.75
    sign = -1.0 if stable else 1.0
    r = 1.0 + sign * abs(r0 - 1.0) * np.exp(-k * t)
    theta = theta0 + 1.20 * t
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def arrow_on_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pos: float, color: str, lw: float = 1.0) -> None:
    idx = int(np.clip(pos * (len(x) - 1), 1, len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx - 1], y[idx - 1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def annulus_background(ax: plt.Axes) -> None:
    outer = plt.Circle((0, 0), 1.55, fc="#F7F5F0", ec="#C9C1B8", lw=1.2, zorder=0)
    inner = plt.Circle((0, 0), 0.65, fc="white", ec="#C9C1B8", lw=1.0, zorder=1)
    ax.add_patch(outer)
    ax.add_patch(inner)
    ax.text(-1.72, 1.52, r"$\Omega_1$", fontsize=14, color="#444444")


def stable_cycle_fig(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.patch.set_facecolor("white")
    draw_axes(ax)
    annulus_background(ax)

    # Stable limit cycle
    cycle = plt.Circle((0, 0), 1.0, fill=False, ec="#CC4638", lw=3.0)
    ax.add_patch(cycle)

    trajectories = [
        (1.42, 0.3, "#2AA198"),
        (1.33, 1.3, "#2AA198"),
        (1.24, 2.35, "#2AA198"),
        (0.78, 0.2, "#C77C2D"),
        (0.86, 1.2, "#C77C2D"),
        (0.92, 2.2, "#C77C2D"),
    ]
    for r0, th0, color in trajectories:
        x, y = spiral_to_cycle(r0, stable=True, theta0=th0, tmax=9.5, n=1200)
        ax.plot(x, y, color=color, lw=1.7, alpha=0.95)
        arrow_on_curve(ax, x, y, 0.40, color, lw=1.0)
        arrow_on_curve(ax, x, y, 0.72, color, lw=1.0)

    # Annotation.
    ax.text(-1.55, 1.86, r"Si $\nabla\cdot(BF)$ no cambia de signo", fontsize=10.7, color="#333333")
    ax.text(-1.55, 1.66, r"en una región anular, existe al menos", fontsize=10.7, color="#333333")
    ax.text(-1.55, 1.46, r"un ciclo límite", fontsize=10.7, color="#333333")
    ax.text(0.70, -1.72, r"Ciclo límite estable", fontsize=13.2, color="#333333")
    ax.text(1.02, 1.06, r"$\gamma$", fontsize=14, color="#CC4638")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def unstable_cycle_fig(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    fig.patch.set_facecolor("white")
    draw_axes(ax)
    annulus_background(ax)

    # Unstable limit cycle
    cycle = plt.Circle((0, 0), 1.0, fill=False, ec="#CC4638", lw=3.0, ls="--")
    ax.add_patch(cycle)

    trajectories = [
        (1.30, 0.2, "#2AA198"),
        (1.22, 1.05, "#2AA198"),
        (1.15, 2.2, "#2AA198"),
        (0.80, 0.5, "#7A3E9D"),
        (0.86, 1.4, "#7A3E9D"),
        (0.92, 2.6, "#7A3E9D"),
    ]
    for r0, th0, color in trajectories:
        x, y = spiral_from_cycle(r0, stable=False, theta0=th0, tmax=8.5, n=1200)
        ax.plot(x, y, color=color, lw=1.7, alpha=0.95)
        arrow_on_curve(ax, x, y, 0.42, color, lw=1.0)
        arrow_on_curve(ax, x, y, 0.72, color, lw=1.0)

    # Add a second faint copy to echo the hand-drawn repetition in the photo.
    cycle2 = plt.Circle((0.12, -0.12), 1.0, fill=False, ec="#CC4638", lw=1.6, alpha=0.35)
    ax.add_patch(cycle2)

    ax.text(-1.55, 1.86, r"Región anular $\Omega_1$", fontsize=10.7, color="#333333")
    ax.text(-1.55, 1.66, r"$\nabla\cdot(BF)$ no cambia de signo", fontsize=10.7, color="#333333")
    ax.text(-1.55, 1.46, r"Entonces existe al menos un ciclo límite", fontsize=10.7, color="#333333")
    ax.text(0.62, -1.72, r"Ciclo límite inestable", fontsize=13.2, color="#333333")
    ax.text(1.02, 1.06, r"$\gamma$", fontsize=14, color="#CC4638")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out1 = OUT_DIR / "fig_43_bendixson_dulac_ciclo_estable_04Jul26.png"
    out2 = OUT_DIR / "fig_44_bendixson_dulac_ciclo_inestable_04Jul26.png"
    stable_cycle_fig(out1)
    unstable_cycle_fig(out2)
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
