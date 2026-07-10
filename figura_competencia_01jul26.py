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


def draw_axes(ax: plt.Axes, xlim=(-0.1, 1.8), ylim=(-0.1, 1.8)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0), arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0))
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]), arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0))
    ax.text(xlim[1] - 0.08, -0.14, r"$u$", fontsize=13)
    ax.text(-0.12, ylim[1] - 0.04, r"$v$", fontsize=13)


def rk4_trajectory(field, y0, tmax=26.0, n=1200):
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


def arrow_on_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pos: float, color: str, lw=1.0) -> None:
    idx = int(np.clip(pos * (len(x) - 1), 1, len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx - 1], y[idx - 1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def field_factory(alpha: float, beta: float, rho: float = 1.0):
    def field(state: np.ndarray) -> np.ndarray:
        u, v = state
        return np.array([u * (1.0 - u - alpha * v), rho * v * (1.0 - v - beta * u)])

    return field


def plot_regime_coexistence(path: Path) -> None:
    alpha = 0.60
    beta = 0.50
    rho = 1.0
    field = field_factory(alpha, beta, rho)
    u_star = (1.0 - alpha) / (1.0 - alpha * beta)
    v_star = (1.0 - beta) / (1.0 - alpha * beta)

    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    fig.patch.set_facecolor("white")
    draw_axes(ax)

    u = np.linspace(0.0, 1.75, 500)
    v_u = (1.0 - u) / alpha
    v_v = 1.0 - beta * u
    ax.plot(np.zeros_like(u), u, color="#4F6DFF", lw=2.1, ls=(0, (6, 4)))
    ax.plot(u, np.zeros_like(u), color="#4F6DFF", lw=2.1)
    ax.plot(u, v_u, color="#D1495B", lw=2.2)
    ax.plot(u, v_v, color="#C77C2D", lw=2.2)

    grid_u = np.linspace(0.02, 1.75, 43)
    grid_v = np.linspace(0.02, 1.75, 43)
    U, V = np.meshgrid(grid_u, grid_v)
    FU = U * (1.0 - U - alpha * V)
    FV = rho * V * (1.0 - V - beta * U)
    ax.streamplot(U, V, FU, FV, density=1.05, color="#96A9FF", linewidth=0.9, arrowsize=1.0, arrowstyle="->")

    initials = [
        (0.12, 1.55),
        (1.55, 1.45),
        (0.16, 0.20),
        (1.40, 0.22),
        (0.62, 1.20),
        (1.10, 0.82),
        (0.35, 1.40),
        (1.55, 0.60),
    ]
    for y0 in initials:
        tu, tv = rk4_trajectory(field, y0, tmax=26.0, n=900)
        ax.plot(tu, tv, color="#2AA198", lw=1.2, alpha=0.95)
        arrow_on_curve(ax, tu, tv, 0.60, "#2AA198", lw=0.95)

    ax.plot(0, 0, "ko", ms=4)
    ax.plot(0, 1, "ko", ms=4)
    ax.plot(1, 0, "ko", ms=4)
    ax.plot(u_star, v_star, "ko", ms=4.5)
    ax.text(0.06, 0.10, r"$E_1=(0,0)$", fontsize=11)
    ax.text(-0.18, 1.05, r"$E_2=(0,1)$", fontsize=11)
    ax.text(1.03, -0.16, r"$E_3=(1,0)$", fontsize=11)
    ax.text(u_star + 0.05, v_star + 0.06, r"$E^*$", fontsize=12)

    ax.text(0.06, 1.70, r"$0<\alpha,\beta<1$", fontsize=13, color="#AA2C2C")
    ax.text(0.56, 1.72, r"$\dot u>0$", fontsize=10.5, color="#0B6E4F")
    ax.text(1.08, 1.22, r"$\dot u<0$", fontsize=10.5, color="#0B6E4F")
    ax.text(1.08, 0.50, r"$\dot v>0$", fontsize=10.5, color="#7A3E9D")
    ax.text(0.30, 0.50, r"$\dot v<0$", fontsize=10.5, color="#7A3E9D")
    ax.text(1.18, 0.08, r"$u=0$", fontsize=10.5, color="#4F6DFF")
    ax.text(1.27, 0.58, r"$1-v-\beta u=0$", fontsize=10.5, color="#C77C2D")
    ax.text(0.58, 0.85, r"$1-u-\alpha v=0$", fontsize=10.5, color="#D1495B")
    ax.text(0.82, 1.62, r"Completar el estudio geométrico", fontsize=12, color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_regime_bistability(path: Path) -> None:
    alpha = 1.35
    beta = 1.25
    rho = 1.0
    field = field_factory(alpha, beta, rho)
    u_star = (1.0 - alpha) / (1.0 - alpha * beta)
    v_star = (1.0 - beta) / (1.0 - alpha * beta)

    fig, ax = plt.subplots(figsize=(8.3, 6.2))
    fig.patch.set_facecolor("white")
    draw_axes(ax)

    u = np.linspace(0.0, 1.75, 500)
    v_u = (1.0 - u) / alpha
    v_v = 1.0 - beta * u
    ax.plot(np.zeros_like(u), u, color="#4F6DFF", lw=2.1, ls=(0, (6, 4)))
    ax.plot(u, np.zeros_like(u), color="#4F6DFF", lw=2.1)
    ax.plot(u, v_u, color="#D1495B", lw=2.2)
    ax.plot(u, v_v, color="#C77C2D", lw=2.2)

    grid_u = np.linspace(0.02, 1.75, 43)
    grid_v = np.linspace(0.02, 1.75, 43)
    U, V = np.meshgrid(grid_u, grid_v)
    FU = U * (1.0 - U - alpha * V)
    FV = rho * V * (1.0 - V - beta * U)
    ax.streamplot(U, V, FU, FV, density=1.05, color="#96A9FF", linewidth=0.9, arrowsize=1.0, arrowstyle="->")

    initials = [
        (0.10, 1.45),
        (1.65, 1.50),
        (0.22, 0.18),
        (1.45, 0.20),
        (0.72, 0.58),
        (0.60, 1.25),
        (1.25, 0.72),
        (1.55, 0.35),
        (0.30, 1.60),
    ]
    for y0 in initials:
        tu, tv = rk4_trajectory(field, y0, tmax=28.0, n=1100)
        ax.plot(tu, tv, color="#C0392B", lw=1.2, alpha=0.93)
        arrow_on_curve(ax, tu, tv, 0.58, "#C0392B", lw=0.95)

    ax.plot(0, 0, "ko", ms=4)
    ax.plot(0, 1, "ko", ms=4)
    ax.plot(1, 0, "ko", ms=4)
    ax.plot(u_star, v_star, "ko", ms=4.5)
    ax.text(0.06, 0.10, r"$E_1=(0,0)$", fontsize=11)
    ax.text(-0.18, 1.05, r"$E_2=(0,1)$", fontsize=11)
    ax.text(1.03, -0.16, r"$E_3=(1,0)$", fontsize=11)
    ax.text(u_star + 0.05, v_star + 0.06, r"$E^*$", fontsize=12)

    ax.text(0.06, 1.70, r"$1<\alpha,\beta$", fontsize=13, color="#AA2C2C")
    ax.text(0.48, 1.50, r"$\dot u>0$", fontsize=10.5, color="#0B6E4F")
    ax.text(1.08, 1.10, r"$\dot u<0$", fontsize=10.5, color="#0B6E4F")
    ax.text(1.02, 0.50, r"$\dot v>0$", fontsize=10.5, color="#7A3E9D")
    ax.text(0.25, 0.56, r"$\dot v<0$", fontsize=10.5, color="#7A3E9D")
    ax.text(1.24, 0.08, r"$u=0$", fontsize=10.5, color="#4F6DFF")
    ax.text(1.29, 0.66, r"$1-v-\beta u=0$", fontsize=10.5, color="#C77C2D")
    ax.text(0.56, 0.82, r"$1-u-\alpha v=0$", fontsize=10.5, color="#D1495B")
    ax.text(0.82, 1.62, r"Completar el estudio geométrico", fontsize=12, color="#333333")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out1 = OUT_DIR / "fig_41_competencia_01Jul26_regimen1.png"
    out2 = OUT_DIR / "fig_42_competencia_01Jul26_regimen2.png"
    plot_regime_coexistence(out1)
    plot_regime_bistability(out2)
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
