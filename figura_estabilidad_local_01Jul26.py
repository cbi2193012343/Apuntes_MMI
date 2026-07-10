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
        "axes.titlesize": 16,
        "axes.labelsize": 12,
    }
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figuras"
OUT_DIR.mkdir(exist_ok=True)


def draw_axes(ax: plt.Axes, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
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
    ax.text(xlim[1] - 0.12, -0.18, r"$u$", fontsize=13)
    ax.text(-0.15, ylim[1] - 0.08, r"$v$", fontsize=13)


def rk4_trajectory(field, y0, tmax=12.0, n=1200):
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
    return y[:, 0], y[:, 1]


def arrow_on_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pos: float, color: str, lw=1.0) -> None:
    idx = int(np.clip(pos * (len(x) - 1), 1, len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx - 1], y[idx - 1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def normalized_stream(ax: plt.Axes, A: np.ndarray, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), density=1.1) -> None:
    u = np.linspace(xlim[0], xlim[1], 41)
    v = np.linspace(ylim[0], ylim[1], 41)
    U, V = np.meshgrid(u, v)
    FU = A[0, 0] * U + A[0, 1] * V
    FV = A[1, 0] * U + A[1, 1] * V
    S = np.hypot(FU, FV)
    S = np.where(S == 0, 1.0, S)
    ax.streamplot(
        U,
        V,
        FU,
        FV,
        density=density,
        color="#90A7FF",
        linewidth=0.9,
        arrowsize=1.0,
        arrowstyle="->",
    )


def plot_trajectories(ax: plt.Axes, A: np.ndarray, initials, color="#2A9D8F", tmax=12.0) -> None:
    def field(y):
        return A @ y

    for y0 in initials:
        x, y = rk4_trajectory(field, y0, tmax=tmax, n=1300)
        ax.plot(x, y, color=color, lw=1.4, alpha=0.97)
        arrow_on_curve(ax, x, y, 0.58, color=color, lw=1.0)


def save_node_stable(path: Path) -> None:
    # Construct a diagonalizable sink with two real negative eigenvalues.
    P = np.array([[1.0, 1.0], [0.95, -0.55]])
    D = np.diag([-2.0, -0.65])
    A = P @ D @ np.linalg.inv(P)
    eigvals, eigvecs = np.linalg.eig(A)

    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    fig.patch.set_facecolor("white")
    draw_axes(ax)
    normalized_stream(ax, A, density=1.0)
    initials = [
        (-1.8, -1.5),
        (-1.7, 1.6),
        (1.7, 1.5),
        (1.6, -1.6),
        (-1.1, 0.5),
        (0.6, -1.2),
        (1.2, 0.25),
    ]
    plot_trajectories(ax, A, initials)

    # Eigenvector directions.
    colors = ["#D1495B", "#C77C2D"]
    for i in range(2):
        v = np.real(eigvecs[:, i])
        v = v / np.linalg.norm(v)
        s = np.linspace(-2.3, 2.3, 2)
        ax.plot(s * v[0], s * v[1], ls="--", lw=1.35, color=colors[i])
        ax.text(1.55 * v[0], 1.55 * v[1], rf"$v_{i+1}$", color=colors[i], fontsize=12)

    ax.scatter([0], [0], s=22, color="black", zorder=5)
    ax.text(-1.92, 1.88, r"$\Delta>0$", fontsize=13, color="#333333")
    ax.text(-1.92, 1.62, r"$\lambda_1<0,\ \lambda_2<0$", fontsize=13, color="#333333")
    ax.text(0.78, -1.95, r"Nodo estable", fontsize=14, color="#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_degenerate_node(path: Path) -> None:
    # Jordan block with a repeated negative eigenvalue.
    J = np.array([[-1.0, 1.0], [0.0, -1.0]])
    P = np.array([[1.0, 0.45], [0.35, 1.0]])
    A = P @ J @ np.linalg.inv(P)
    eigvals, eigvecs = np.linalg.eig(A)

    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    fig.patch.set_facecolor("white")
    draw_axes(ax)
    normalized_stream(ax, A, density=1.0)
    initials = [
        (-1.8, -1.4),
        (-1.6, 1.2),
        (1.8, 1.5),
        (1.6, -1.5),
        (0.9, 1.2),
        (-0.4, 1.7),
        (1.2, -0.8),
    ]
    plot_trajectories(ax, A, initials, color="#C0392B", tmax=15.0)

    v = np.real(eigvecs[:, 0])
    v = v / np.linalg.norm(v)
    s = np.linspace(-2.3, 2.3, 2)
    ax.plot(s * v[0], s * v[1], ls="--", lw=1.45, color="#1F7A8C")
    ax.text(1.55 * v[0], 1.55 * v[1], r"$v_1$", color="#1F7A8C", fontsize=12)

    ax.scatter([0], [0], s=22, color="black", zorder=5)
    ax.text(-1.92, 1.88, r"$\Delta=0$", fontsize=13, color="#333333")
    ax.text(-1.92, 1.62, r"$\lambda_{1,2}<0$  multiplicidad 2", fontsize=13, color="#333333")
    ax.text(0.72, -1.95, r"Nodo degenerado estable", fontsize=14, color="#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_stable_spiral(path: Path) -> None:
    A = np.array([[-0.25, -1.0], [1.0, -0.25]])

    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    fig.patch.set_facecolor("white")
    draw_axes(ax)
    normalized_stream(ax, A, density=1.15)
    initials = [
        (-1.8, -1.5),
        (-1.7, 1.5),
        (1.7, 1.4),
        (1.4, -1.6),
        (-0.2, 1.8),
        (1.8, 0.2),
        (-1.3, 0.1),
    ]
    plot_trajectories(ax, A, initials, color="#2A9D8F", tmax=22.0)

    ax.scatter([0], [0], s=22, color="black", zorder=5)
    ax.text(-1.92, 1.88, r"$\Delta<0$", fontsize=13, color="#333333")
    ax.text(-1.92, 1.62, r"$\Re(\lambda)<0$", fontsize=13, color="#333333")
    ax.text(0.92, -1.95, r"Espiral estable", fontsize=14, color="#333333")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out1 = OUT_DIR / "fig_38_nodo_estable_01Jul26.png"
    out2 = OUT_DIR / "fig_39_nodo_degenerado_estable_01Jul26.png"
    out3 = OUT_DIR / "fig_40_espiral_estable_01Jul26.png"
    save_node_stable(out1)
    save_degenerate_node(out2)
    save_stable_spiral(out3)
    print(out1)
    print(out2)
    print(out3)


if __name__ == "__main__":
    main()
