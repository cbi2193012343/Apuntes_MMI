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


def hide_frame(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_axes(ax: plt.Axes, xlim=(-0.1, 1.8), ylim=(-0.1, 1.9)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    hide_frame(ax)
    ax.annotate(
        "",
        xy=(xlim[1], 0),
        xytext=(xlim[0], 0),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(0, ylim[1]),
        xytext=(0, ylim[0]),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
    )
    ax.text(xlim[1] - 0.08, -0.14, r"$u$", fontsize=13)
    ax.text(-0.12, ylim[1] - 0.04, r"$v$", fontsize=13)


def rk4_trajectory(field, y0, tmax=30.0, n=1200):
    t = np.linspace(0.0, tmax, n)
    y = np.zeros((n, 2), dtype=float)
    y[0] = np.asarray(y0, dtype=float)
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        yi = y[i]
        k1 = field(yi)
        k2 = field(yi + 0.5 * dt * k1)
        k3 = field(yi + 0.5 * dt * k2)
        k4 = field(yi + dt * k3)
        y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[i + 1] = np.maximum(y[i + 1], 1e-6)
    return y[:, 0], y[:, 1]


def arrow_on_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pos: float, color, lw=1.1) -> None:
    idx = int(np.clip(pos * (len(x) - 1), 1, len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx - 1], y[idx - 1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def competition_field(alpha: float, beta: float, rho: float):
    def field(state: np.ndarray) -> np.ndarray:
        u, v = state
        return np.array([u * (1.0 - u - alpha * v), rho * v * (1.0 - v - beta * u)])

    return field


def base_competition_plot(ax: plt.Axes, alpha: float, beta: float, rho: float, *, show_stream: bool) -> None:
    u_star = (1.0 - alpha) / (1.0 - alpha * beta)
    v_star = (1.0 - beta) / (1.0 - alpha * beta)
    field = competition_field(alpha, beta, rho)

    draw_axes(ax)

    # Nullclines.
    u_line = np.linspace(0.0, 1.75, 500)
    v_u = (1.0 - u_line) / alpha
    v_v = 1.0 - beta * u_line

    ax.plot(np.zeros_like(u_line), u_line, color="#4F6DFF", lw=2.1, ls=(0, (6, 4)))
    ax.plot(u_line, np.zeros_like(u_line), color="#4F6DFF", lw=2.1)
    ax.plot(u_line, v_u, color="#D1495B", lw=2.2)
    ax.plot(u_line, v_v, color="#C77C2D", lw=2.2)

    # Intercepts of nullclines.
    ax.plot(0, 1 / alpha, "o", ms=4, color="#D1495B")
    ax.plot(1 / beta, 0, "o", ms=4, color="#C77C2D")
    ax.text(0.04, 1 / alpha + 0.07, r"$\left(0,\frac{1}{\alpha}\right)$", fontsize=10.5, color="#D1495B")
    ax.text(1 / beta - 0.05, -0.14, r"$\left(\frac{1}{\beta},0\right)$", fontsize=10.5, color="#C77C2D")

    # Equilibria.
    ax.plot(0, 0, "ko", ms=4)
    ax.plot(0, 1, "ko", ms=4)
    ax.plot(1, 0, "ko", ms=4)
    ax.plot(u_star, v_star, "ko", ms=4.5)
    ax.text(0.06, 0.10, r"$E_1=(0,0)$", fontsize=11)
    ax.text(-0.18, 1.05, r"$E_2=(0,1)$", fontsize=11)
    ax.text(1.03, -0.16, r"$E_3=(1,0)$", fontsize=11)
    ax.text(u_star + 0.05, v_star + 0.06, r"$E^*$", fontsize=12)

    # Sign regions.
    ax.text(0.18, 0.22, r"$\dot u>0,\ \dot v>0$", fontsize=10.5, color="#0B6E4F")
    ax.text(0.16, 1.22, r"$\dot u>0,\ \dot v<0$", fontsize=10.5, color="#7A3E9D")
    ax.text(1.12, 0.48, r"$\dot u<0,\ \dot v>0$", fontsize=10.5, color="#7A3E9D")
    ax.text(1.07, 1.35, r"$\dot u<0,\ \dot v<0$", fontsize=10.5, color="#C0392B")

    # Optional stream plot and sample trajectories.
    if show_stream:
        grid_u = np.linspace(0.02, 1.75, 42)
        grid_v = np.linspace(0.02, 1.75, 42)
        U, V = np.meshgrid(grid_u, grid_v)
        FU = U * (1.0 - U - alpha * V)
        FV = rho * V * (1.0 - V - beta * U)
        ax.streamplot(
            U,
            V,
            FU,
            FV,
            density=1.05,
            color="#7890FF",
            linewidth=0.92,
            arrowsize=1.0,
            arrowstyle="->",
        )

        trajectories = [
            (0.12, 1.55),
            (1.55, 1.45),
            (0.16, 0.20),
            (1.40, 0.22),
            (0.62, 1.20),
            (1.10, 0.82),
        ]
        for y0 in trajectories:
            tu, tv = rk4_trajectory(field, y0, tmax=26.0, n=900)
            ax.plot(tu, tv, color="#2AA198", lw=1.2, alpha=0.95)
            arrow_on_curve(ax, tu, tv, 0.60, "#2AA198", lw=0.95)


def save_fig_competencia_bosquejo(path: Path) -> None:
    alpha = 0.60
    beta = 0.50
    rho = 1.0
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    fig.patch.set_facecolor("white")
    base_competition_plot(ax, alpha, beta, rho, show_stream=False)
    ax.text(0.80, 1.72, r"Bosquejo parcial", fontsize=13, color="#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_competencia_completa(path: Path) -> None:
    alpha = 0.60
    beta = 0.50
    rho = 1.0
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    fig.patch.set_facecolor("white")
    base_competition_plot(ax, alpha, beta, rho, show_stream=True)
    ax.text(0.78, 1.72, r"Retrato completo", fontsize=13, color="#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out1 = OUT_DIR / "fig_36_competencia_bosquejo_29Jun26.png"
    out2 = OUT_DIR / "fig_37_competencia_completa_29Jun26.png"
    save_fig_competencia_bosquejo(out1)
    save_fig_competencia_completa(out2)
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
