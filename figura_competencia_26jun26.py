from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 15,
        "axes.labelsize": 12,
    }
)


def hide_frame(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_axes(ax: plt.Axes, xlim=(-0.1, 1.8), ylim=(-0.1, 1.8)) -> None:
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


def main() -> None:
    A = 0.60
    B = 0.50
    rho = 1.0

    def field(state: np.ndarray) -> np.ndarray:
        u, v = state
        return np.array([u * (1.0 - u - A * v), rho * v * (1.0 - v - B * u)])

    u_star = (1.0 - A) / (1.0 - A * B)
    v_star = (1.0 - B) / (1.0 - A * B)

    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    fig.patch.set_facecolor("white")
    draw_axes(ax)

    u = np.linspace(0.02, 1.75, 42)
    v = np.linspace(0.02, 1.75, 42)
    U, V = np.meshgrid(u, v)
    FU = U * (1.0 - U - A * V)
    FV = rho * V * (1.0 - V - B * U)

    ax.streamplot(
        U,
        V,
        FU,
        FV,
        density=1.1,
        color="#7890FF",
        linewidth=0.95,
        arrowsize=1.0,
        arrowstyle="->",
    )

    u_line = np.linspace(0.0, 1.75, 400)
    v_u = (1.0 - u_line) / A
    v_v = 1.0 - B * u_line

    ax.plot(u_line, v_u, color="#D1495B", lw=2.1)
    ax.plot(u_line, v_v, color="#C77C2D", lw=2.1)
    ax.plot(np.zeros_like(u_line), u_line, color="#4F6DFF", lw=2.0, ls=(0, (6, 4)))
    ax.plot(u_line, np.zeros_like(u_line), color="#4F6DFF", lw=2.0)

    trajectories = [
        (0.12, 1.45),
        (1.55, 1.40),
        (0.18, 0.20),
        (1.38, 0.22),
        (0.62, 1.18),
    ]
    for y0 in trajectories:
        tu, tv = rk4_trajectory(field, y0, tmax=26.0, n=900)
        ax.plot(tu, tv, color="#2AA198", lw=1.2, alpha=0.95)
        arrow_on_curve(ax, tu, tv, 0.60, "#2AA198", lw=0.95)

    # Equilibria.
    ax.plot(0, 0, "ko", ms=4)
    ax.plot(0, 1, "ko", ms=4)
    ax.plot(1, 0, "ko", ms=4)
    ax.plot(u_star, v_star, "ko", ms=4.5)

    ax.text(0.05, 0.13, r"$E_1=(0,0)$", fontsize=11.5)
    ax.text(-0.18, 1.05, r"$E_2=(0,1)$", fontsize=11.5)
    ax.text(1.03, -0.16, r"$E_3=(1,0)$", fontsize=11.5)
    ax.text(u_star + 0.05, v_star + 0.06, r"$E^*$", fontsize=12.5)

    ax.text(0.92, 1.50, r"$v=\frac{1-u}{A}$", fontsize=12, color="#D1495B")
    ax.text(1.20, 0.92, r"$v=1-Bu$", fontsize=12, color="#C77C2D")
    ax.text(0.80, 0.08, r"$u=0$", fontsize=11, color="#4F6DFF")
    ax.text(0.02, 1.60, r"$v=0$", fontsize=11, color="#4F6DFF")

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "figuras")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_35_competencia_26Jun26.png")
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
