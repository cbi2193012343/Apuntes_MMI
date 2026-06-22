"""Figures for the flow and conjugacy notes.

The module generates schematic figures that match the handwritten pages:

- a saddle-type phase portrait for a hyperbolic equilibrium;
- a single-orbit sketch showing the flow map \varphi_t(x);
- a composition sketch showing \varphi_s(\varphi_t(x)) = \varphi_{s+t}(x).

The PNGs are meant to be included later in the LaTeX notes.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def _hide_frame(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_center_axes(ax: plt.Axes, xlim, ylim, color="#222222") -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.annotate(
        "",
        xy=(xlim[1], 0),
        xytext=(xlim[0], 0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )
    ax.annotate(
        "",
        xy=(0, ylim[1]),
        xytext=(0, ylim[0]),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
    )
    _hide_frame(ax)


def _arrow_on_curve(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pos: float, color, lw=1.4) -> None:
    """Draw a small arrow tangentially along a curve."""
    idx = int(np.clip(pos * (len(x) - 1), 1, len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx - 1], y[idx - 1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )


def _cubic_bezier(p0, p1, p2, p3, n=400):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    pts = (
        ((1 - t) ** 3) * p0
        + 3 * ((1 - t) ** 2) * t * p1
        + 3 * (1 - t) * (t**2) * p2
        + (t**3) * p3
    )
    return pts[:, 0], pts[:, 1]


def save_fig_center_periodic(path: str) -> None:
    """Phase portrait of Y' = AY with A = [[0,-1],[1,0]]: a center."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _draw_center_axes(ax, (-2.4, 2.4), (-2.4, 2.4))

    grid = np.linspace(-2.3, 2.3, 35)
    X, Y = np.meshgrid(grid, grid)
    U = -Y
    V = X
    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.0,
        color="#4F6DFF",
        linewidth=1.0,
        arrowsize=1.0,
        arrowstyle="->",
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    for radius in (0.45, 0.8, 1.15, 1.5, 1.85, 2.15):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot(x, y, color="#2E6BFF", lw=1.7)
        _arrow_on_curve(ax, x, y, 0.12, "#2E6BFF", lw=1.05)

    ax.plot(0, 0, "ko", ms=4)
    ax.text(2.18, -0.18, r"$y_1$", fontsize=13)
    ax.text(-0.16, 2.2, r"$y_2$", fontsize=13)
    ax.text(0.05, 0.95, r"$Y'=AY$", transform=ax.transAxes, fontsize=13)
    ax.text(0.05, 0.08, "A = [[0, -1], [1, 0]]", transform=ax.transAxes, fontsize=12)
    ax.text(0.61, 0.10, r"órbitas cerradas", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_spiral_no_example(path: str) -> None:
    """Phase portrait of the nonlinear no-example: outward spirals."""
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    _draw_center_axes(ax, (-1.65, 1.65), (-1.65, 1.65))

    grid = np.linspace(-1.55, 1.55, 35)
    X, Y = np.meshgrid(grid, grid)
    R2 = X**2 + Y**2
    U = -Y + X * R2
    V = X + Y * R2
    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.05,
        color="#4F6DFF",
        linewidth=0.95,
        arrowsize=1.0,
        arrowstyle="->",
    )

    def spiral_curve(x0: float, y0: float, tmax_scale: float = 0.86, n: int = 520):
        r0 = float(np.hypot(x0, y0))
        theta0 = float(np.arctan2(y0, x0))
        blowup = 1.0 / max(2.0 * r0 * r0, 1e-8)
        tmax = min(1.9, tmax_scale * blowup)
        t = np.linspace(0.0, tmax, n)
        denom = np.sqrt(np.maximum(1.0 - 2.0 * r0 * r0 * t, 1e-7))
        r = r0 / denom
        theta = theta0 + t
        return r * np.cos(theta), r * np.sin(theta)

    starts = [
        (0.42, 0.0),
        (0.0, 0.42),
        (-0.42, 0.0),
        (0.0, -0.42),
        (0.58, 0.22),
        (-0.52, 0.28),
    ]
    for x0, y0 in starts:
        x, y = spiral_curve(x0, y0)
        ax.plot(x, y, color="#2E6BFF", lw=1.6)
        _arrow_on_curve(ax, x, y, 0.55, "#2E6BFF", lw=1.05)

    ax.add_patch(plt.Circle((0, 0), 0.42, fill=False, ls=(0, (6, 5)), ec="#D1495B", lw=1.6))
    ax.plot(0, 0, "ko", ms=4)
    ax.text(1.46, -0.18, r"$x_1$", fontsize=13)
    ax.text(-0.15, 1.48, r"$x_2$", fontsize=13)
    ax.text(0.05, 0.95, r"$X'=F(X)$", transform=ax.transAxes, fontsize=13)
    ax.text(0.06, 0.10, r"$\dot r=r^3$", transform=ax.transAxes, fontsize=12, color="#8B0000")
    ax.text(0.72, 0.12, r"$\dot\theta=1$", transform=ax.transAxes, fontsize=12, color="#8B0000")
    ax.text(0.06, 0.18, r"espirales salientes", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_isoclinas_schema(path: str) -> None:
    """Schematic decomposition of X' = F(X) into its horizontal and vertical components."""
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    fig.patch.set_facecolor("white")

    ax = axs[0]
    ax.set_xlim(-0.6, 4.1)
    ax.set_ylim(-0.5, 3.4)
    ax.set_aspect("equal", adjustable="box")
    _hide_frame(ax)
    ax.annotate(
        "",
        xy=(4.0, 0),
        xytext=(-0.5, 0),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(0, 3.3),
        xytext=(0, -0.4),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
    )
    ax.text(3.92, -0.16, r"$x$", fontsize=13)
    ax.text(-0.16, 3.18, r"$y$", fontsize=13)

    px, py = 1.45, 1.22
    fx, fy = 1.35, 0.92
    ax.plot(px, py, "ko", ms=4)
    ax.text(px + 0.10, py + 0.08, r"$P=(x,y)$", fontsize=12)

    ax.plot([px, px], [0, py], ls="--", lw=1.0, color="#999999")
    ax.plot([0, px], [py, py], ls="--", lw=1.0, color="#999999")

    ax.add_patch(
        FancyArrowPatch(
            (px, py),
            (px + fx, py + fy),
            arrowstyle="->",
            mutation_scale=16,
            lw=2.2,
            color="#2E6BFF",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (px, py),
            (px + fx, py),
            arrowstyle="->",
            mutation_scale=14,
            lw=1.9,
            color="#D1495B",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (px + fx, py),
            (px + fx, py + fy),
            arrowstyle="->",
            mutation_scale=14,
            lw=1.9,
            color="#2AA198",
        )
    )
    ax.text(px + 0.54, py - 0.22, r"$f(x,y)$", fontsize=12, color="#D1495B")
    ax.text(px + fx + 0.08, py + 0.38, r"$g(x,y)$", fontsize=12, color="#2AA198")
    ax.text(px + 0.37, py + 0.63, r"$F(X)$", fontsize=13, color="#2E6BFF")

    axs[1].axis("off")
    axs[1].text(0.02, 0.93, r"$X' = F(X)$", fontsize=17)
    axs[1].text(0.02, 0.80, r"$F:\mathbb{R}^2 \to \mathbb{R}^2$ de clase $C^1$", fontsize=14)
    axs[1].text(0.02, 0.61, r"$X(t)=\binom{x(t)}{y(t)}$", fontsize=15)
    axs[1].text(0.02, 0.48, r"$F(X)=\binom{f(x,y)}{g(x,y)}$", fontsize=15)
    axs[1].text(0.02, 0.28, r"$x'=f(x,y)\ \Rightarrow\ x'=0\ \Leftrightarrow\ f(x,y)=0\ \Rightarrow\ I_1$", fontsize=12)
    axs[1].text(0.02, 0.15, r"$y'=g(x,y)\ \Rightarrow\ y'=0\ \Leftrightarrow\ g(x,y)=0\ \Rightarrow\ I_2$", fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_isoclinas_phase(path: str) -> None:
    """Phase portrait showing a representative pair of nullclines I_1 and I_2."""
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    _draw_center_axes(ax, (-2.4, 2.4), (-2.0, 2.15))

    def field(z):
        x, y = float(z[0]), float(z[1])
        return np.array([x - y, 1.0 - x * x - y], dtype=float)

    x = np.linspace(-2.25, 2.25, 43)
    y = np.linspace(-1.9, 2.05, 43)
    X, Y = np.meshgrid(x, y)
    U = X - Y
    V = 1.0 - X**2 - Y
    speed = np.hypot(U, V)

    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.05,
        color="#4F6DFF",
        linewidth=0.95,
        arrowsize=1.0,
        arrowstyle="->",
    )

    xs = np.linspace(-2.25, 2.25, 500)
    i1 = xs
    i2 = 1.0 - xs**2
    ax.plot(xs, i1, color="#D1495B", lw=2.0)
    ax.plot(xs, i2, color="#2AA198", lw=2.0, ls=(0, (6, 4)))
    _arrow_on_curve(ax, xs, i1, 0.80, "#D1495B", lw=1.0)
    _arrow_on_curve(ax, xs, i2, 0.34, "#2AA198", lw=1.0)

    def traj(y0):
        return _rk4_trajectory(field, y0, tmax=4.0, n=360)

    trajectories = [
        (-1.8, 1.2),
        (-1.25, -0.8),
        (-0.8, 1.55),
        (0.1, -0.55),
        (0.95, 1.05),
        (1.65, -0.55),
    ]
    for y0 in trajectories:
        tx, ty = traj(y0)
        ax.plot(tx, ty, color="#2E6BFF", lw=1.5, alpha=0.9)
        _arrow_on_curve(ax, tx, ty, 0.58, "#2E6BFF", lw=1.0)

    # Highlight the intersection points of the nullclines.
    roots = [(-1.0 + np.sqrt(5.0)) / 2.0, (-1.0 - np.sqrt(5.0)) / 2.0]
    for xr in roots:
        yr = xr
        ax.plot(xr, yr, "ko", ms=3.6)

    ax.text(2.1, -0.18, r"$x$", fontsize=13)
    ax.text(-0.15, 2.02, r"$y$", fontsize=13)
    ax.text(0.03, 0.95, r"campo de fases con isoclinas", transform=ax.transAxes, fontsize=12)
    ax.text(0.55, 0.86, r"$I_1:\ x'=0$", transform=ax.transAxes, fontsize=12, color="#D1495B")
    ax.text(0.13, 0.20, r"$I_2:\ y'=0$", transform=ax.transAxes, fontsize=12, color="#2AA198")
    ax.text(0.05, 0.08, r"$x'=x-y,\quad y'=1-x^2-y$", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.72, 0.10, r"espacio de trayectorias", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_isoclinas_signs(path: str) -> None:
    """Sign chart associated with the nullclines I_1 and I_2."""
    fig, ax = plt.subplots(figsize=(7.3, 6.0))
    _draw_center_axes(ax, (-2.4, 2.4), (-1.95, 2.1))

    xs = np.linspace(-2.25, 2.25, 500)
    i1 = xs
    i2 = 1.0 - xs**2

    # Soft background hints for the signs of x' and y' with respect to the nullclines.
    ax.fill_between(xs, i2, 2.1, color="#BFE6EA", alpha=0.18)
    ax.fill_between(xs, -1.95, i2, color="#D9F0D6", alpha=0.18)

    ax.plot(xs, i1, color="#D1495B", lw=2.1)
    ax.plot(xs, i2, color="#2AA198", lw=2.1, ls=(0, (6, 4)))
    _arrow_on_curve(ax, xs, i1, 0.78, "#D1495B", lw=1.0)
    _arrow_on_curve(ax, xs, i2, 0.35, "#2AA198", lw=1.0)

    # Representative sample points in the four sign regions.
    sample_points = [
        (-1.6, 1.3, r"$x'<0,\ y'<0$"),
        (1.2, 1.45, r"$x'<0,\ y'<0$"),
        (-1.5, -1.3, r"$x'>0,\ y'>0$"),
        (1.7, -0.8, r"$x'>0,\ y'>0$"),
        (0.25, 0.15, r"$x'>0,\ y'<0$"),
        (-0.5, 0.85, r"$x'<0,\ y'>0$"),
    ]
    for px, py, label in sample_points:
        ax.plot(px, py, "ko", ms=2.7)
        ax.text(px + 0.08, py + 0.08, label, fontsize=10)

    ax.text(1.18, 1.80, r"$x'<0$", fontsize=12, color="#9A3B49")
    ax.text(0.92, -1.55, r"$x'>0$", fontsize=12, color="#9A3B49")
    ax.text(-1.95, 1.55, r"$y'<0$", fontsize=12, color="#268A7F")
    ax.text(-1.95, -1.40, r"$y'>0$", fontsize=12, color="#268A7F")

    ax.text(2.0, -0.17, r"$x$", fontsize=13)
    ax.text(-0.16, 1.93, r"$y$", fontsize=13)
    ax.text(0.05, 0.94, r"diagrama de signos", transform=ax.transAxes, fontsize=12)
    ax.text(0.72, 0.87, r"$I_1$", transform=ax.transAxes, fontsize=12, color="#D1495B")
    ax.text(0.82, 0.60, r"$I_2$", transform=ax.transAxes, fontsize=12, color="#2AA198")
    ax.text(0.04, 0.10, r"$I_1:\ x'=0$", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.18, 0.10, r"$I_2:\ y'=0$", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_competition_nullclines(path: str) -> None:
    """Nullclines for x' = x(1-x-y), y' = y(1-x)."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    _draw_center_axes(ax, (-0.8, 2.05), (-0.8, 1.8))

    xs = np.linspace(-0.75, 2.0, 500)
    ys = np.linspace(-0.75, 1.75, 500)

    # Nullclines.
    ax.plot(xs, np.zeros_like(xs), color="#B24B5A", lw=2.1)
    ax.plot(np.ones_like(ys), ys, color="#B24B5A", lw=2.1)
    ax.plot(xs, 1.0 - xs, color="#355CFF", lw=2.2)
    ax.plot(np.zeros_like(ys), ys, color="#355CFF", lw=2.1, ls=(0, (5, 4)))

    # Label the associated isoclines.
    ax.text(1.28, 1.34, r"$x'=0$", fontsize=13, color="#355CFF")
    ax.text(1.05, 0.46, r"$y'=0$", fontsize=13, color="#B24B5A")
    ax.text(-0.12, 1.62, r"$x=0$", fontsize=12, color="#355CFF")
    ax.text(0.20, -0.18, r"$y=0$", fontsize=12, color="#B24B5A")
    ax.text(0.72, 0.88, r"$y=1-x$", fontsize=12, color="#355CFF")
    ax.text(1.02, 0.12, r"$x=1$", fontsize=12, color="#B24B5A")

    # Equilibria.
    ax.plot(0, 0, "ko", ms=4)
    ax.plot(1, 0, "ko", ms=4)
    ax.text(0.06, 0.08, r"$(0,0)$", fontsize=11)
    ax.text(1.05, 0.08, r"$(1,0)$", fontsize=11)

    # Local direction hints on the axes.
    ax.annotate("", xy=(0.65, 0), xytext=(0.12, 0), arrowprops=dict(arrowstyle="->", color="#B24B5A", lw=1.4))
    ax.annotate("", xy=(1.65, 0), xytext=(1.15, 0), arrowprops=dict(arrowstyle="->", color="#B24B5A", lw=1.4))
    ax.annotate("", xy=(0, 0.75), xytext=(0, 0.20), arrowprops=dict(arrowstyle="->", color="#355CFF", lw=1.4))
    ax.text(0.05, 0.95, r"esquema de isoclinas", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.90, r"modelo: $x'=x(1-x-y),\ y'=y(1-x)$", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_competition_signs(path: str) -> None:
    """Sign regions for the competition system with the main nullclines."""
    fig, ax = plt.subplots(figsize=(8.1, 6.6))
    _draw_center_axes(ax, (-0.8, 2.2), (-0.8, 1.95))

    xs = np.linspace(-0.75, 2.1, 500)
    ys = np.linspace(-0.75, 1.9, 500)

    # Background emphasis on the positive quadrant.
    ax.fill_between([0, 2.2], [0, 0], [1.95, 1.95], color="#F6F5D7", alpha=0.22)

    # Nullclines.
    ax.plot(xs, np.zeros_like(xs), color="#B24B5A", lw=2.1)
    ax.plot(np.ones_like(ys), ys, color="#B24B5A", lw=2.1)
    ax.plot(xs, 1.0 - xs, color="#355CFF", lw=2.2)
    ax.plot(np.zeros_like(ys), ys, color="#355CFF", lw=2.1, ls=(0, (5, 4)))

    # Representative points and sign labels.
    samples = [
        (-0.55, 1.35, r"$x'<0,\ y'<0$"),
        (-0.45, -0.45, r"$x'>0,\ y'<0$"),
        (0.28, 0.28, r"$x'>0,\ y'>0$"),
        (0.28, 1.05, r"$x'<0,\ y'>0$"),
        (1.36, 0.30, r"$x'<0,\ y'<0$"),
        (1.52, 1.40, r"$x'<0,\ y'<0$"),
        (1.55, -0.32, r"$x'>0,\ y'<0$"),
    ]
    for px, py, label in samples:
        ax.plot(px, py, "ko", ms=3.1)
        ax.text(px + 0.06, py + 0.07, label, fontsize=11)

    # Invariant positive axes.
    ax.annotate("", xy=(0.62, 0), xytext=(0.12, 0), arrowprops=dict(arrowstyle="->", color="#B24B5A", lw=1.4))
    ax.annotate("", xy=(1.55, 0), xytext=(1.12, 0), arrowprops=dict(arrowstyle="->", color="#B24B5A", lw=1.4))
    ax.annotate("", xy=(0, 0.70), xytext=(0, 0.18), arrowprops=dict(arrowstyle="->", color="#355CFF", lw=1.4))

    ax.text(0.05, 0.94, r"signos de $x'$ y $y'$", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.88, r"ejes positivos invariantes", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(1.53, 1.68, r"$x'=0$", fontsize=12, color="#355CFF")
    ax.text(1.02, 0.52, r"$y'=0$", fontsize=12, color="#B24B5A")
    ax.text(0.65, 1.15, r"$x'>0,\ y'>0$", fontsize=12, color="#2C5E39")
    ax.text(0.62, 1.48, r"$x'<0,\ y'>0$", fontsize=12, color="#2C5E39")
    ax.text(1.34, 0.92, r"$x'<0,\ y'<0$", fontsize=12, color="#7A3C45")

    ax.text(1.95, -0.18, r"$x$", fontsize=13)
    ax.text(-0.15, 1.83, r"$y$", fontsize=13)
    ax.text(0.05, 0.10, r"$x'=x(1-x-y),\qquad y'=y(1-x)$", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_predator_prey_schema(path: str) -> None:
    """Stylized predator-prey habitat sketch with prey, predators and interactions."""
    fig, ax = plt.subplots(figsize=(8.9, 5.8))
    ax.set_xlim(-2.5, 2.8)
    ax.set_ylim(-1.8, 1.95)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    _hide_frame(ax)

    rng = np.random.default_rng(7)

    def scatter_cloud(center, spread, n, color, alpha=0.92, size=10):
        pts = rng.normal(loc=center, scale=spread, size=(n, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, alpha=alpha, linewidths=0)

    prey_center = (-0.25, 0.10)
    predator_center = (0.95, 0.78)
    interaction_center = (0.30, 0.35)

    # Main clouds.
    scatter_cloud(prey_center, (0.58, 0.36), 190, "#2E6BFF", alpha=0.85, size=9)
    scatter_cloud(predator_center, (0.32, 0.24), 140, "#D1495B", alpha=0.88, size=10)
    scatter_cloud(interaction_center, (0.22, 0.15), 70, "#2AA198", alpha=0.65, size=8)

    # Soft interaction rings.
    ax.add_patch(Circle(prey_center, 0.72, fill=False, ls=(0, (3, 4)), ec="#2E6BFF", lw=1.8, alpha=0.70))
    ax.add_patch(Circle(predator_center, 0.50, fill=False, ls=(0, (3, 4)), ec="#D1495B", lw=1.8, alpha=0.72))
    ax.add_patch(Circle(interaction_center, 0.26, fill=False, ls=(0, (2, 3)), ec="#2AA198", lw=1.6, alpha=0.82))

    # Suggested movement / interaction.
    ax.add_patch(
        FancyArrowPatch(
            (-0.10, 0.55),
            (0.70, 0.70),
            connectionstyle="arc3,rad=0.25",
            arrowstyle="->",
            mutation_scale=18,
            lw=1.8,
            color="#2AA198",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.72, 0.52),
            (0.02, 0.18),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="->",
            mutation_scale=18,
            lw=1.8,
            color="#2AA198",
        )
    )

    # Left-side labels matching the board.
    ax.text(-2.15, 1.10, r"$P$ presas", fontsize=15, color="#2E6BFF")
    ax.text(-2.15, 0.54, r"$D$ depredadores", fontsize=15, color="#D1495B")
    ax.text(-2.15, -0.10, r"Interacciones", fontsize=15, color="#2AA198")
    ax.text(-2.15, 1.62, r"Depredador-presa", fontsize=16, color="#444444")

    # Small habitat cue.
    ax.text(1.55, -1.38, "poblaciones distribuidas en su hábitat", fontsize=11, color="#555555")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_invariant_octant(path: str) -> None:
    """3D schematic for an invariant positive octant and coordinate planes."""
    fig = plt.figure(figsize=(8.8, 6.9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(0, 1.55)
    ax.set_zlim(0, 1.25)
    ax.set_box_aspect((1.05, 1.05, 0.85))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Make panes nearly invisible so the planes stand out.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass

    grid = np.linspace(0.0, 1.45, 2)
    U, V = np.meshgrid(grid, grid)

    # Coordinate planes.
    ax.plot_surface(np.zeros_like(U), U, V, color="#4F6DFF", alpha=0.18, linewidth=0.0, shade=False)
    ax.plot_surface(U, np.zeros_like(U), V, color="#D1495B", alpha=0.18, linewidth=0.0, shade=False)
    ax.plot_surface(U, V, np.zeros_like(U), color="#2AA198", alpha=0.18, linewidth=0.0, shade=False)

    # Bounding edges for the first octant sketch.
    edge = 1.38
    for xs, ys, zs in [
        ([0, edge], [0, 0], [0, 0]),
        ([0, 0], [0, edge], [0, 0]),
        ([0, 0], [0, 0], [0, edge]),
        ([edge, edge], [0, edge], [0, 0]),
        ([edge, edge], [0, 0], [0, edge]),
        ([0, edge], [edge, edge], [0, 0]),
        ([0, 0], [edge, edge], [0, edge]),
        ([0, edge], [0, 0], [edge, edge]),
        ([0, 0], [0, edge], [edge, edge]),
    ]:
        ax.plot(xs, ys, zs, color="#666666", lw=0.8, alpha=0.55)

    # Axes arrows.
    ax.quiver(0, 0, 0, 1.52, 0, 0, color="#222222", arrow_length_ratio=0.07, linewidth=1.6)
    ax.quiver(0, 0, 0, 0, 1.52, 0, color="#222222", arrow_length_ratio=0.07, linewidth=1.6)
    ax.quiver(0, 0, 0, 0, 0, 1.28, color="#222222", arrow_length_ratio=0.07, linewidth=1.6)
    ax.text(1.60, 0.00, 0.00, r"$x_1$", fontsize=13)
    ax.text(0.00, 1.60, 0.00, r"$x_2$", fontsize=13)
    ax.text(0.00, 0.00, 1.33, r"$x_3$", fontsize=13)

    # Flow arrows inside the octant.
    starts = np.array(
        [
            [0.18, 0.18, 0.12],
            [0.35, 0.55, 0.18],
            [0.62, 0.22, 0.55],
            [0.22, 0.88, 0.42],
            [0.90, 0.82, 0.28],
        ]
    )
    vecs = np.array(
        [
            [0.28, 0.15, 0.16],
            [0.18, 0.20, 0.22],
            [0.18, 0.10, 0.18],
            [0.10, 0.20, 0.16],
            [0.14, 0.14, 0.12],
        ]
    )
    for s, v in zip(starts, vecs):
        ax.quiver(
            s[0],
            s[1],
            s[2],
            v[0],
            v[1],
            v[2],
            color="#355CFF",
            arrow_length_ratio=0.25,
            linewidth=1.5,
        )

    # Plane labels and octant note.
    ax.text(0.04, 1.20, 1.08, r"$x_1=0$", color="#4F6DFF", fontsize=12)
    ax.text(1.10, 0.04, 1.06, r"$x_2=0$", color="#D1495B", fontsize=12)
    ax.text(1.05, 1.00, 0.04, r"$x_3=0$", color="#2AA198", fontsize=12)
    fig.text(0.73, 0.91, r"plano $x_1=0$ invariante", fontsize=10.5, color="#4F6DFF")
    fig.text(0.73, 0.86, r"plano $x_2=0$ invariante", fontsize=10.5, color="#D1495B")
    fig.text(0.73, 0.81, r"plano $x_3=0$ invariante", fontsize=10.5, color="#2AA198")
    fig.text(0.73, 0.76, r"el primer octante es invariante", fontsize=11.5, color="#444444")

    fig.subplots_adjust(left=0.00, right=0.72, bottom=0.00, top=0.98)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _draw_lv_frame(ax: plt.Axes, xlim=(-0.55, 2.55), ylim=(-0.55, 2.55)) -> None:
    """Draw axes for the predator-prey phase plane and label P/D."""
    _draw_center_axes(ax, xlim, ylim)
    ax.text(xlim[1] - 0.18, -0.17, r"$P$", fontsize=13)
    ax.text(-0.15, ylim[1] - 0.04, r"$D$", fontsize=13)


def _plot_lv_nullclines(ax: plt.Axes, p_star: float = 1.0, d_star: float = 1.0) -> None:
    xs = np.linspace(-0.5, 2.5, 500)
    ys = np.linspace(-0.5, 2.5, 500)
    ax.plot(xs, np.full_like(xs, d_star), color="#D1495B", lw=2.2)
    ax.plot(np.full_like(ys, p_star), ys, color="#355CFF", lw=2.2)


def save_fig_predator_prey_stage1(path: str) -> None:
    """Initial predator-prey sketch: invariant axes and nullclines."""
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    _draw_lv_frame(ax)
    _plot_lv_nullclines(ax)

    ax.plot(0, 0, "ko", ms=4)
    ax.plot(1, 1, "ko", ms=4)
    ax.text(0.05, 0.94, "analisis 1: isoclinas nulas", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.88, r"$P=0$ y $D=0$ son ejes invariantes", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(1.08, 1.04, r"$D=\alpha/\beta$", fontsize=12, color="#D1495B")
    ax.text(1.02, 1.28, r"$P=\gamma/\delta$", fontsize=12, color="#355CFF")
    ax.text(0.06, 0.08, r"$E_1=(0,0)$", transform=ax.transAxes, fontsize=11)
    ax.text(1.11, 1.16, r"$E^*$", fontsize=12)
    ax.text(0.05, 0.03, r"$P'=P(\alpha-\beta D),\quad D'=D(\delta P-\gamma)$", transform=ax.transAxes, fontsize=10.5, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_predator_prey_stage2(path: str) -> None:
    """Sign regions for the predator-prey system in the biologically relevant phase plane."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    _draw_lv_frame(ax)
    _plot_lv_nullclines(ax)

    ax.text(0.03, 0.95, "analisis de signos", transform=ax.transAxes, fontsize=12)
    ax.text(0.03, 0.89, "biologicamente, solo el primer cuadrante", transform=ax.transAxes, fontsize=11, color="#444444")

    samples = [
        (0.35, 1.55, r"$P'<0,\ D'<0$"),
        (1.45, 1.55, r"$P'<0,\ D'>0$"),
        (0.35, 0.35, r"$P'>0,\ D'<0$"),
        (1.45, 0.35, r"$P'>0,\ D'>0$"),
    ]
    for px, py, label in samples:
        ax.plot(px, py, "ko", ms=3.1)
        ax.text(px + 0.06, py + 0.07, label, fontsize=11)

    arrows = [
        ((0.55, 0.60), (0.82, 0.42)),
        ((1.35, 0.60), (1.65, 0.88)),
        ((0.55, 1.35), (0.25, 1.10)),
        ((1.35, 1.35), (1.12, 1.66)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=16,
                lw=1.6,
                color="#2AA198",
            )
        )

    ax.text(1.53, 1.68, r"$P'<0$", fontsize=12, color="#9A3B49")
    ax.text(1.02, 0.52, r"$D'=0$", fontsize=12, color="#D1495B")
    ax.text(0.65, 1.15, r"$D'>0$", fontsize=12, color="#2C5E39")
    ax.text(0.62, 1.48, r"$P'<0$", fontsize=12, color="#9A3B49")
    ax.text(1.34, 0.92, r"$P'>0,\ D'<0$", fontsize=12, color="#7A3C45")

    ax.text(1.08, 1.08, r"$D=\alpha/\beta$", fontsize=12, color="#D1495B")
    ax.text(1.02, 1.28, r"$P=\gamma/\delta$", fontsize=12, color="#355CFF")
    ax.text(0.06, 0.08, r"regiones de signos", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_predator_prey_stage3(path: str) -> None:
    """Phase portrait of the normalized Lotka-Volterra system in the first quadrant."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _draw_lv_frame(ax, xlim=(-0.12, 2.55), ylim=(-0.12, 2.55))

    x = np.linspace(0.05, 2.45, 70)
    y = np.linspace(0.05, 2.45, 70)
    X, Y = np.meshgrid(x, y)
    U = X * (1.0 - Y)
    V = Y * (X - 1.0)

    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.1,
        color="#7A8CFF",
        linewidth=0.9,
        arrowsize=1.0,
        arrowstyle="->",
    )

    H = X - np.log(X) + Y - np.log(Y)
    levels = [2.10, 2.25, 2.50, 2.85, 3.30, 4.0]
    ax.contour(X, Y, H, levels=levels, colors="#355CFF", linewidths=1.35, alpha=0.92)

    ax.axvline(1.0, color="#355CFF", lw=2.0)
    ax.axhline(1.0, color="#D1495B", lw=2.0)
    ax.plot(0, 0, "ko", ms=4)
    ax.plot(1, 1, "ko", ms=4)

    ax.text(0.05, 0.95, "dinamica local en el primer cuadrante", transform=ax.transAxes, fontsize=12)
    ax.text(0.06, 0.88, r"$E_1$ es una silla; $E^*$ organiza las orbitas", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.09, 0.15, r"$E_1$", fontsize=12)
    ax.text(1.06, 1.04, r"$E^*$", fontsize=12)
    ax.text(1.05, 1.24, r"$D=\alpha/\beta$", fontsize=11, color="#D1495B")
    ax.text(1.18, 0.96, r"$P=\gamma/\delta$", fontsize=11, color="#355CFF")
    ax.text(0.04, 0.05, r"$P'=P(\alpha-\beta D),\quad D'=D(\delta P-\gamma)$", transform=ax.transAxes, fontsize=10.5, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _rk4_trajectory(field, y0, tmax=3.0, n=400):
    """Integrate a 2D autonomous system with fixed-step RK4."""
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


def _plot_stream_panel(
    ax: plt.Axes,
    xlim,
    ylim,
    u_fun,
    v_fun,
    trajectories,
    panel_title: str,
    eq_label: str,
    axis_x_label: str,
    axis_y_label: str,
    circle_label: str,
    circle_radius: float = 0.8,
    stream_color: str = "#355CFF",
    traj_color: str = "#4F6DFF",
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    _hide_frame(ax)

    x = np.linspace(xlim[0], xlim[1], 35)
    y = np.linspace(ylim[0], ylim[1], 35)
    X, Y = np.meshgrid(x, y)
    U = u_fun(X, Y)
    V = v_fun(X, Y)

    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.0,
        color=stream_color,
        linewidth=0.95,
        arrowsize=0.95,
        arrowstyle="->",
    )

    for traj_fn, y0 in trajectories:
        tx, ty = traj_fn(y0)
        ax.plot(tx, ty, color=traj_color, lw=1.45, alpha=0.9)
        _arrow_on_curve(ax, tx, ty, 0.62, traj_color, lw=1.0)

    ax.add_patch(
        Circle((0, 0), circle_radius, fill=False, ls=(0, (6, 5)), lw=1.5, ec="#D1495B")
    )
    ax.plot(0, 0, "ko", ms=3)
    ax.text(xlim[1] - 0.18, -0.18, axis_x_label, fontsize=13)
    ax.text(-0.15, ylim[1] - 0.04, axis_y_label, fontsize=13)
    ax.text(0.03, 0.95, panel_title, transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(0.06, 0.12, circle_label, transform=ax.transAxes, color="#D1495B", fontsize=12)
    ax.text(0.02, 0.02, eq_label, transform=ax.transAxes, fontsize=11, color="#444444")


def save_fig_hg_conjugacy_local(path: str) -> None:
    """Side-by-side local portraits around X0 for the nonlinear and linearized systems."""
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.patch.set_facecolor("white")

    def linear_traj(y0):
        t = np.linspace(0.0, 2.8, 260)
        x = y0[0] * np.exp(-t)
        y = y0[1] * np.exp(-2.0 * t)
        return x, y

    def nonlinear_field(z):
        return np.array([-z[0] + z[0] ** 2, -2.0 * z[1] + z[0] ** 2], dtype=float)

    def nonlinear_traj(y0):
        return _rk4_trajectory(nonlinear_field, y0, tmax=3.0, n=300)

    _plot_stream_panel(
        axs[0],
        (-2.1, 2.1),
        (-2.1, 2.1),
        lambda X, Y: -X,
        lambda X, Y: -2.0 * Y,
        [
            (linear_traj, (-1.7, 1.2)),
            (linear_traj, (-1.5, -0.7)),
            (linear_traj, (-0.9, 1.3)),
            (linear_traj, (1.6, 1.0)),
            (linear_traj, (1.2, -0.5)),
        ],
        r"$Y'=DF(X_0)Y$",
        r"$H(x_0)$",
        r"$y_1$",
        r"$y_2$",
        r"$V_0$",
        circle_radius=1.05,
    )

    _plot_stream_panel(
        axs[1],
        (-1.1, 1.1),
        (-1.1, 1.1),
        lambda X, Y: -X + X**2,
        lambda X, Y: -2.0 * Y + X**2,
        [
            (nonlinear_traj, (-0.9, 0.5)),
            (nonlinear_traj, (-0.75, -0.55)),
            (nonlinear_traj, (-0.45, 0.8)),
            (nonlinear_traj, (0.7, 0.55)),
            (nonlinear_traj, (0.85, -0.35)),
        ],
        r"$X'=F(X)$",
        r"$x_0$",
        r"$x_1$",
        r"$x_2$",
        r"$U_{x_0}$",
        circle_radius=0.78,
    )

    arrow = FancyArrowPatch(
        (0.45, 0.63),
        (0.55, 0.63),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=20,
        lw=2.2,
        color="#2AA198",
    )
    fig.add_artist(arrow)
    fig.text(0.50, 0.665, r"$H$", ha="center", va="bottom", fontsize=17, color="#2AA198")

    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_saddle_portrait(path: str) -> None:
    """Hyperbolic saddle with diagonals as separatrices."""
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    _draw_center_axes(ax, (-2.6, 2.6), (-2.6, 2.6))

    grid = np.linspace(-2.4, 2.4, 31)
    X, Y = np.meshgrid(grid, grid)

    # Rotated saddle: x' = y, y' = x.
    U = Y
    V = X
    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.0,
        color="#355CFF",
        linewidth=1.0,
        arrowsize=1.0,
        arrowstyle="->",
    )

    # Stable / unstable eigendirections.
    u = np.linspace(-2.4, 2.4, 200)
    ax.plot(u, u, color="#D1495B", lw=2.0)
    ax.plot(u, -u, color="#D1495B", lw=2.0)

    # A few selected trajectories to make the structure easier to read.
    for c in (0.35, 1.0, 1.65):
        y = np.linspace(-2.2, 2.2, 400)
        x = np.sqrt(y**2 + c)
        ax.plot(x, y, color="#4F6DFF", lw=1.1, alpha=0.55)
        ax.plot(-x, y, color="#4F6DFF", lw=1.1, alpha=0.55)
        _arrow_on_curve(ax, x, y, 0.32, "#4F6DFF", lw=1.0)
        _arrow_on_curve(ax, -x, y, 0.68, "#4F6DFF", lw=1.0)

    ax.text(2.72, -0.12, r"$x$", fontsize=14)
    ax.text(-0.12, 2.72, r"$y$", fontsize=14)
    ax.plot(0, 0, "ko", ms=3)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_fig_flow_segment(path: str) -> None:
    """Single orbit sketch with an initial point x and a later point phi_t(x)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _draw_center_axes(ax, (-2.9, 2.7), (-0.6, 2.2))

    p0 = (-2.2, 0.15)
    p1 = (-1.2, 0.55)
    p2 = (0.0, 1.55)
    p3 = (2.1, 1.25)
    x, y = _cubic_bezier(p0, p1, p2, p3)

    ax.plot(x, y, color="#2E6BFF", lw=2.0)
    _arrow_on_curve(ax, x, y, 0.25, "#2E6BFF", lw=1.2)
    _arrow_on_curve(ax, x, y, 0.65, "#2E6BFF", lw=1.2)

    ax.plot([p0[0], p3[0]], [p0[1], p3[1]], "o", color="#1f1f1f", ms=4)
    ax.text(p0[0] - 0.12, p0[1] - 0.16, r"$x$", fontsize=13)
    ax.text(p3[0] - 0.34, p3[1] + 0.05, r"$\varphi_t(x)$", fontsize=13)
    ax.text(0.12, 1.53, r"$\varphi_t$", color="#1f1f1f", fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_fig_flow_composition(path: str) -> None:
    """Flow composition sketch: phi_s(phi_t(x)) = phi_{s+t}(x)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    _draw_center_axes(ax, (-2.9, 3.1), (-0.6, 2.3))

    p0 = (-2.35, 0.20)
    p1 = (-1.35, 0.65)
    p2 = (-0.15, 1.55)
    p3 = (1.10, 1.60)
    p4 = (2.45, 1.05)

    # Split the orbit in two pieces to emphasize the composition.
    x1, y1 = _cubic_bezier(p0, p1, p2, p3, n=260)
    x2, y2 = _cubic_bezier(p3, (1.35, 1.55), (1.85, 1.25), p4, n=220)
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])

    ax.plot(x, y, color="#2E6BFF", lw=2.0)
    _arrow_on_curve(ax, x, y, 0.20, "#2E6BFF", lw=1.2)
    _arrow_on_curve(ax, x, y, 0.52, "#2E6BFF", lw=1.2)
    _arrow_on_curve(ax, x, y, 0.82, "#2E6BFF", lw=1.2)

    # Mark the three relevant points.
    ax.plot([p0[0], p3[0], p4[0]], [p0[1], p3[1], p4[1]], "o", color="#1f1f1f", ms=4)
    ax.text(p0[0] - 0.10, p0[1] - 0.16, r"$x$", fontsize=13)
    ax.text(p3[0] - 0.55, p3[1] + 0.08, r"$\varphi_t(x)$", fontsize=13)
    ax.text(p4[0] - 0.95, p4[1] - 0.08, r"$\varphi_s(\varphi_t(x))$", fontsize=13)
    ax.text(0.02, 1.78, r"$\varphi_{s+t}(x)$", fontsize=12, color="#1f1f1f")
    ax.text(-0.55, 0.88, r"$t$", fontsize=12, color="#8B0000")
    ax.text(1.67, 1.35, r"$s$", fontsize=12, color="#8B0000")
    ax.text(-0.25, 0.42, r"$\varphi_s(\varphi_t(x))=\varphi_{s+t}(x)$", fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_fig_first_integral_schema(path: str) -> None:
    """Conceptual sketch of a first integral and the orthogonality relation."""
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    _draw_center_axes(ax, (-2.5, 2.5), (-2.25, 2.25))

    grid = np.linspace(-2.35, 2.35, 45)
    X, Y = np.meshgrid(grid, grid)
    H = X**2 + 0.7 * Y**2
    U = -1.4 * Y
    V = 2.0 * X
    ax.streamplot(
        X,
        Y,
        U,
        V,
        density=1.0,
        color="#4F6DFF",
        linewidth=0.95,
        arrowsize=1.0,
        arrowstyle="->",
    )

    levels = [0.45, 0.90, 1.45, 2.10, 2.95, 3.90]
    ax.contour(X, Y, H, levels=levels, colors="#2AA198", linewidths=1.15, alpha=0.9)

    c0 = 1.75
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    x_orbit = np.sqrt(c0) * np.cos(theta)
    y_orbit = np.sqrt(c0 / 0.7) * np.sin(theta)
    ax.plot(x_orbit, y_orbit, color="#2E6BFF", lw=2.1)
    _arrow_on_curve(ax, x_orbit, y_orbit, 0.12, "#2E6BFF", lw=1.1)

    t0 = 0.88
    x0 = np.sqrt(c0) * np.cos(t0)
    y0 = np.sqrt(c0 / 0.7) * np.sin(t0)
    ax.plot(x0, y0, "ko", ms=4)
    ax.text(x0 + 0.08, y0 - 0.08, r"$X(t)$", fontsize=12)

    grad = np.array([2.0 * x0, 1.4 * y0], dtype=float)
    flow = np.array([-1.4 * y0, 2.0 * x0], dtype=float)

    def _draw_vector(vec, color, label, offset=(0.0, 0.0), label_shift=(0.0, 0.0)):
        norm = float(np.hypot(vec[0], vec[1]))
        if norm < 1e-9:
            return
        vec = vec * (0.72 / norm)
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x0 + vec[0], y0 + vec[1]),
                arrowstyle="->",
                mutation_scale=16,
                lw=2.0,
                color=color,
            )
        )
        ax.text(
            x0 + vec[0] + label_shift[0],
            y0 + vec[1] + label_shift[1],
            label,
            fontsize=12,
            color=color,
        )

    _draw_vector(flow, "#355CFF", r"$F(X(t))$", label_shift=(0.05, -0.05))
    _draw_vector(grad, "#D1495B", r"$\nabla H(X(t))$", label_shift=(-0.42, 0.08))

    ax.text(0.57, 0.93, r"integral primera: $H(X(t))=\mathrm{cte.}$", transform=ax.transAxes, fontsize=12)
    ax.text(0.57, 0.85, r"$0=\dfrac{d}{dt}H(X(t))=\nabla H(X(t))\cdot F(X(t))$", transform=ax.transAxes, fontsize=11)
    ax.text(0.57, 0.77, r"$\nabla H(X(t))\perp F(X(t))$", transform=ax.transAxes, fontsize=12, color="#444444")
    ax.text(0.06, 0.10, r"las trayectorias viven sobre niveles de $H$", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(1.92, 1.82, r"$H(x,y)=c$", fontsize=12, color="#2AA198")
    ax.text(2.10, -0.18, r"$x$", fontsize=13)
    ax.text(-0.15, 2.03, r"$y$", fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_logistic_harvest(path: str) -> None:
    """Two-panel schematic for a logistic growth model with constant harvest."""
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.2))
    fig.patch.set_facecolor("white")

    alpha = 1.2
    k = 4.0
    H0 = 1.0
    p = np.linspace(0.0, 4.2, 600)
    growth = alpha * p * (1.0 - p / k)
    net = growth - H0
    roots = np.roots([-alpha / k, alpha, -H0])
    roots = np.sort(roots[np.isreal(roots)].real)

    # Panel 1: growth curve and harvest level.
    ax = axs[0]
    _draw_center_axes(ax, (-0.05, 4.25), (-0.45, 1.55))
    ax.plot(p, growth, color="#355CFF", lw=2.2)
    ax.axhline(H0, color="#D1495B", lw=2.0, ls=(0, (6, 4)))
    ax.fill_between(p, H0, growth, where=growth >= H0, color="#BFE6D3", alpha=0.45)
    ax.fill_between(p, growth, H0, where=growth < H0, color="#F4D6D6", alpha=0.35)
    for idx, r in enumerate(roots):
        ax.plot(r, H0, "ko", ms=4)
        label = r"$p_-$" if idx == 0 else r"$p_+$"
        ax.text(r + 0.05, H0 + 0.07, label, fontsize=11)
    ax.axvline(k / 2.0, color="#999999", lw=1.0, ls=":")
    ax.text(k / 2.0 + 0.04, 1.28, r"$p=\frac{k}{2}$", fontsize=11, color="#666666")
    ax.text(0.05, 0.92, r"$f(p)=\alpha p\left(1-\frac{p}{k}\right)$", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.84, r"cosecha constante: $H$", transform=ax.transAxes, fontsize=11, color="#D1495B")
    ax.text(0.05, 0.12, r"$p'>0$ donde $f(p)>H$", transform=ax.transAxes, fontsize=11, color="#2C5E39")
    ax.text(0.05, 0.06, r"$p'<0$ donde $f(p)<H$", transform=ax.transAxes, fontsize=11, color="#7A3C45")
    ax.text(4.04, -0.18, r"$p$", fontsize=13)
    ax.text(-0.15, 1.38, r"$\dot p$", fontsize=13)

    # Panel 2: shifted net growth p' = f(p) - H.
    ax = axs[1]
    _draw_center_axes(ax, (-0.05, 4.25), (-1.25, 0.95))
    ax.plot(p, net, color="#2E6BFF", lw=2.2)
    ax.axhline(0.0, color="#222222", lw=1.0)
    ax.fill_between(p, 0.0, net, where=net >= 0, color="#BFE6D3", alpha=0.45)
    ax.fill_between(p, net, 0.0, where=net < 0, color="#F4D6D6", alpha=0.35)
    for r in roots:
        ax.plot(r, 0.0, "ko", ms=4)
    ax.text(0.06, 0.92, r"$\dot p=\alpha p\left(1-\frac{p}{k}\right)-H$", transform=ax.transAxes, fontsize=12)
    ax.text(0.06, 0.84, r"desplazamiento vertical por $H$", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.11, 0.64, r"$\dot p<0$", transform=ax.transAxes, fontsize=12, color="#7A3C45")
    ax.text(0.42, 0.78, r"$\dot p>0$", transform=ax.transAxes, fontsize=12, color="#2C5E39")
    ax.text(0.83, 0.64, r"$\dot p<0$", transform=ax.transAxes, fontsize=12, color="#7A3C45")
    ax.text(1.97, -0.18, r"$p$", fontsize=13)
    ax.text(-0.15, 0.72, r"$\dot p$", fontsize=13)
    ax.text(0.05, 0.05, r"las intersecciones marcan el cambio de signo", transform=ax.transAxes, fontsize=11, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_predator_prey_first_integral_derivation(path: str) -> None:
    """Flowchart-style derivation of the first integral for the predator-prey system."""
    fig = plt.figure(figsize=(12.4, 7.3))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    def box(x, y, text, color="#222222", fs=16, fc="#F7F7F7", ec="#888888"):
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=fs,
            color=color,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.0),
        )

    ax.text(
        0.05,
        0.94,
        "¿Cómo podemos obtener más información?",
        transform=ax.transAxes,
        fontsize=16,
        color="#2AA198",
        weight="bold",
    )
    box(0.05, 0.83, r"$P' = P(\alpha-\beta D)$", color="#355CFF", fs=17)
    box(0.05, 0.74, r"$D' = D(-\gamma+\delta P)$", color="#355CFF", fs=17)
    ax.text(0.31, 0.78, r"$\Downarrow$", transform=ax.transAxes, fontsize=28, color="#666666")
    box(0.38, 0.80, r"$\dfrac{dD}{dP}=\dfrac{D(-\gamma+\delta P)}{P(\alpha-\beta D)}$", color="#8B0000", fs=16)
    box(0.38, 0.67, r"$\left(\dfrac{-\gamma+\delta P}{P}\right)dP=\left(\dfrac{\alpha-\beta D}{D}\right)dD$", color="#8B0000", fs=15)
    box(0.16, 0.52, r"$\int\left(-\dfrac{\gamma}{P}+\delta\right)dP=\int\left(\dfrac{\alpha}{D}-\beta\right)dD$", color="#8B0000", fs=15)
    box(0.16, 0.40, r"$-\gamma\ln P+\delta P=\alpha\ln D-\beta D + C$", color="#8B0000", fs=16)
    box(0.16, 0.28, r"$H(P,D)=-\gamma\ln P+\delta P+\beta D-\alpha\ln D$", color="#2AA198", fs=16)
    box(0.16, 0.17, r"$H(P,D)=\mathrm{cte.}$", color="#D1495B", fs=17, fc="#FFF6F6", ec="#D1495B")

    ax.text(
        0.58,
        0.17,
        "sus curvas de nivel son órbitas del sistema",
        transform=ax.transAxes,
        fontsize=12,
        color="#444444",
    )
    ax.text(0.58, 0.08, r"En particular, la información global ya no sale del linealizado.", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.90, 0.03, r"$P,D>0$", transform=ax.transAxes, fontsize=12, color="#444444")

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_fig_predator_prey_first_integral_orbits(path: str) -> None:
    """Level-set portrait of the predator-prey first integral in the positive quadrant."""
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    fig.patch.set_facecolor("white")
    _draw_center_axes(ax, (-0.15, 2.75), (-0.15, 2.75))

    p = np.linspace(0.05, 2.65, 180)
    d = np.linspace(0.05, 2.65, 180)
    P, D = np.meshgrid(p, d)
    U = P * (1.0 - D)
    V = D * (P - 1.0)
    ax.streamplot(
        P,
        D,
        U,
        V,
        density=1.1,
        color="#4F6DFF",
        linewidth=0.95,
        arrowsize=1.0,
        arrowstyle="->",
    )

    H = P - np.log(P) + D - np.log(D)
    levels = [2.02, 2.08, 2.18, 2.35, 2.60, 2.95, 3.45]
    ax.contour(P, D, H, levels=levels, colors="#2AA198", linewidths=1.15, alpha=0.95)

    ax.axvline(1.0, color="#D1495B", lw=2.0)
    ax.axhline(1.0, color="#D1495B", lw=2.0)
    ax.plot(0, 0, "ko", ms=3.5)
    ax.plot(1, 1, "ko", ms=4.5)

    ax.text(1.04, 1.05, r"$E^*$", fontsize=12)
    ax.text(0.08, 0.08, r"$E_1$", fontsize=12)
    ax.text(1.05, 2.48, r"$P=\gamma/\delta$", fontsize=11, color="#D1495B")
    ax.text(2.02, 1.05, r"$D=\alpha/\beta$", fontsize=11, color="#D1495B")
    ax.text(0.05, 0.95, r"$H(P,D)=\mathrm{cte.}$", transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.89, r"curvas de nivel de la integral primera", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(0.73, 0.83, r"órbitas cerradas en el primer cuadrante", transform=ax.transAxes, fontsize=11, color="#444444")
    ax.text(2.48, -0.18, r"$P$", fontsize=13)
    ax.text(-0.15, 2.52, r"$D$", fontsize=13)

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_fig_center_periodic("fig_18_sistema_lineal_periodico.png")
    save_fig_spiral_no_example("fig_19_no_ejemplo_espiral.png")
    save_fig_isoclinas_schema("fig_20_isoclinas_esquema.png")
    save_fig_isoclinas_phase("fig_21_isoclinas_retrato.png")
    save_fig_isoclinas_signs("fig_22_isoclinas_signos.png")
    save_fig_competition_nullclines("fig_23_competencia_isoclinas.png")
    save_fig_competition_signs("fig_24_competencia_signos.png")
    save_fig_predator_prey_schema("fig_25_depredador_presa.png")
    save_fig_invariant_octant("fig_26_octante_invariante.png")
    save_fig_predator_prey_stage1("fig_27_depredador_presa_etapa1.png")
    save_fig_predator_prey_stage2("fig_28_depredador_presa_etapa2.png")
    save_fig_predator_prey_stage3("fig_29_depredador_presa_etapa3.png")
    save_fig_first_integral_schema("fig_30_integral_primera_esquema.png")
    save_fig_logistic_harvest("fig_31_cosecha_logistica.png")
    save_fig_predator_prey_first_integral_derivation("fig_32_depredador_presa_integral_derivacion.png")
    save_fig_predator_prey_first_integral_orbits("fig_33_depredador_presa_integral_orbitas.png")
    save_fig_saddle_portrait("fig_14_flujo_silla.png")
    save_fig_flow_segment("fig_15_flujo_orbita.png")
    save_fig_flow_composition("fig_16_flujo_composicion.png")
    save_fig_hg_conjugacy_local("fig_17_hg_conjugacion_local.png")


if __name__ == "__main__":
    main()
