from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

STABLE = "#1f4e79"
UNSTABLE = "#b03a2e"
ACCENT = "#d97706"
SOFT = "#6b7280"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#d9dde3",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


def save_figure(fig: plt.Figure, name: str) -> None:
    pdf_path = FIGS / f"{name}.pdf"
    png_path = FIGS / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#4b5563")
    ax.spines["bottom"].set_color("#4b5563")
    ax.tick_params(colors="#374151")


def plot_masked(ax: plt.Axes, x: np.ndarray, y: np.ndarray, mask: np.ndarray, **kwargs) -> None:
    y_masked = np.where(mask, y, np.nan)
    ax.plot(x, y_masked, **kwargs)


def fig_ej1() -> None:
    a0 = 10.0
    k = 0.5
    t = np.linspace(0.0, 10.0, 500)
    a = a0 * np.exp(-k * t)
    b = a0 * (1.0 - np.exp(-k * t))
    t_half = np.log(2.0) / k

    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    ax.plot(t, a, color=STABLE, linewidth=2.5, label=r"$A(t)=A_0e^{-kt}$")
    ax.plot(t, b, color=ACCENT, linewidth=2.5, label=r"$B(t)=A_0(1-e^{-kt})$")
    ax.axhline(a0, color=SOFT, linestyle="--", linewidth=1.6, label=r"$A(t)+B(t)=A_0$")
    ax.axvline(
        t_half,
        color="#7c3aed",
        linestyle=":",
        linewidth=1.8,
        label=rf"$t_{{1/2}}=\ln 2/k\approx {t_half:.2f}$",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Concentracion")
    ax.set_title("Reaccion de primer orden")
    style_axis(ax)
    ax.legend(loc="center right")
    save_figure(fig, "ej1_AB_primer_orden")


def fig_ej2() -> None:
    a0 = 10.0
    k = 0.1
    t = np.linspace(0.0, 20.0, 500)
    a = a0 / (1.0 + 2.0 * k * a0 * t)
    b = a0 - a
    a_exp = a0 * np.exp(-(2.0 * k * a0) * t / a0)

    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    ax.plot(t, a, color=STABLE, linewidth=2.5, label=r"$A(t)=\dfrac{A_0}{1+2kA_0t}$")
    ax.plot(t, b, color=ACCENT, linewidth=2.5, label=r"$B(t)=A_0-A(t)$")
    ax.axhline(a0, color=SOFT, linestyle="--", linewidth=1.6, label=r"$A(t)+B(t)=A_0$")
    ax.plot(
        t,
        a_exp,
        color=STABLE,
        linestyle=":",
        linewidth=1.9,
        alpha=0.8,
        label="Referencia exponencial",
    )
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Concentracion")
    ax.set_title("Reaccion de segundo orden")
    style_axis(ax)
    ax.legend(loc="center right")
    save_figure(fig, "ej2_AB_segundo_orden")


def fig_ej3_geometrico() -> None:
    r = 1.0
    k = 5.0
    n = np.linspace(-1.0, 7.0, 600)
    f = r * n * (1.0 - n / k)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.2),
        gridspec_kw={"height_ratios": [3.1, 1.6]},
        constrained_layout=True,
    )

    ax0, ax1 = axes
    ax0.plot(n, f, color=STABLE, linewidth=2.6)
    ax0.axhline(0, color="#374151", linewidth=1.0)
    ax0.axvline(0, color="#9ca3af", linewidth=0.9)
    ax0.axvline(k, color="#9ca3af", linewidth=0.9, linestyle="--")
    ax0.scatter([0], [0], s=70, facecolors="white", edgecolors=UNSTABLE, linewidths=2.0, zorder=5)
    ax0.scatter([k], [0], s=70, color=STABLE, zorder=5)
    ax0.text(0.12, 0.55, "inestable", color=UNSTABLE)
    ax0.text(k + 0.12, 0.55, "estable", color=STABLE)
    ax0.text(k / 2.0 - 0.35, r * k / 4.0 + 0.2, r"$N=K/2$", color="#374151")
    ax0.set_xlim(-1, 7)
    ax0.set_ylim(-3.1, 2.1)
    ax0.set_xlabel(r"$N$")
    ax0.set_ylabel(r"$f(N)$")
    ax0.set_title(r"Funcion $f(N)=rN\left(1-\dfrac{N}{K}\right)$ con $r=1$ y $K=5$")
    style_axis(ax0)

    ax1.hlines(0, -1, 7, color="#374151", linewidth=1.3)
    ax1.scatter([0], [0], s=90, facecolors="white", edgecolors=UNSTABLE, linewidths=2.0, zorder=5)
    ax1.scatter([k], [0], s=90, color=STABLE, zorder=5)
    for x0, x1 in [(-0.8, -0.15), (5.8, 5.15)]:
        ax1.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops={"arrowstyle": "->", "color": UNSTABLE, "lw": 2})
    ax1.annotate("", xy=(3.7, 0), xytext=(1.2, 0), arrowprops={"arrowstyle": "->", "color": STABLE, "lw": 2})
    ax1.text(-0.75, 0.18, r"$N<0$", color="#374151")
    ax1.text(1.7, 0.18, r"$0<N<K$", color="#374151")
    ax1.text(5.35, 0.18, r"$N>K$", color="#374151")
    ax1.set_xlim(-1, 7)
    ax1.set_ylim(-0.8, 0.9)
    ax1.set_yticks([])
    ax1.set_xlabel(r"Linea de fase en $N$")
    ax1.grid(False)
    style_axis(ax1)
    save_figure(fig, "ej3_logistico_geometrico")


def fig_ej3_soluciones() -> None:
    r = 1.0
    k = 5.0
    t = np.linspace(0.0, 8.0, 500)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    crecientes = [0.5, 2.0, 4.0]
    decrecientes = [6.0, 8.0, 12.0]
    blue_shades = ["#2563eb", "#1d4ed8", "#1e40af"]
    red_shades = ["#ef4444", "#dc2626", "#991b1b"]

    for n0, color in zip(crecientes, blue_shades):
        n = k / (1.0 + ((k - n0) / n0) * np.exp(-r * t))
        ax.plot(t, n, color=color, linewidth=2.2, label=rf"$N_0={n0:g}$")

    for n0, color in zip(decrecientes, red_shades):
        n = k / (1.0 + ((k - n0) / n0) * np.exp(-r * t))
        ax.plot(t, n, color=color, linewidth=2.2, label=rf"$N_0={n0:g}$")

    ax.axhline(k, color="#111827", linestyle="--", linewidth=1.6, label=r"$N=K$")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$N(t)$")
    ax.set_title("Familia de soluciones del modelo logistico")
    style_axis(ax)
    ax.legend(ncol=3, loc="upper center")
    save_figure(fig, "ej3_logistico_soluciones")


def bifurcation_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color=STABLE, lw=2.5, label="estable"),
        Line2D([0], [0], color=UNSTABLE, lw=2.5, linestyle="--", label="inestable"),
    ]
    ax.legend(handles=handles, loc="upper left")


def base_bif_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], xlabel: str = r"$\mu$") -> None:
    ax.axhline(0, color="#374151", linewidth=1.0)
    ax.axvline(0, color="#9ca3af", linewidth=1.0, linestyle=":")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$y^*$")
    style_axis(ax)


def fig_pitchfork_super(name: str, title: str) -> None:
    mu = np.linspace(-2.0, 2.0, 900)
    branch = np.sqrt(np.clip(mu, 0, None))

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    plot_masked(ax, mu, np.zeros_like(mu), mu <= 0, color=STABLE, linewidth=2.6)
    plot_masked(ax, mu, np.zeros_like(mu), mu >= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    plot_masked(ax, mu, branch, mu >= 0, color=STABLE, linewidth=2.6)
    plot_masked(ax, mu, -branch, mu >= 0, color=STABLE, linewidth=2.6)
    ax.scatter([0], [0], s=34, color="#111827", zorder=5)
    ax.set_title(title)
    base_bif_axis(ax, (-2, 2), (-1.7, 1.7))
    bifurcation_legend(ax)
    save_figure(fig, name)


def fig_transcritical(name: str, title: str) -> None:
    mu = np.linspace(-2.0, 2.0, 900)

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    plot_masked(ax, mu, np.zeros_like(mu), mu <= 0, color=STABLE, linewidth=2.6)
    plot_masked(ax, mu, np.zeros_like(mu), mu >= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    plot_masked(ax, mu, mu, mu <= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    plot_masked(ax, mu, mu, mu >= 0, color=STABLE, linewidth=2.6)
    ax.scatter([0], [0], s=34, color="#111827", zorder=5)
    ax.set_title(title)
    base_bif_axis(ax, (-2, 2), (-2.2, 2.2))
    bifurcation_legend(ax)
    save_figure(fig, name)


def fig_saddle_node() -> None:
    mu = np.linspace(-2.0, 1.0, 900)
    branch = np.sqrt(np.clip(-mu, 0, None))

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    plot_masked(ax, mu, -branch, mu <= 0, color=STABLE, linewidth=2.6)
    plot_masked(ax, mu, branch, mu <= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    ax.scatter([0], [0], s=34, color="#111827", zorder=5)
    ax.set_title("Bifurcacion silla-nodo")
    base_bif_axis(ax, (-2, 1), (-1.7, 1.7))
    bifurcation_legend(ax)
    save_figure(fig, "bif_f_saddle_node")


def fig_pitchfork_sub() -> None:
    mu = np.linspace(-2.0, 2.0, 900)
    branch = np.sqrt(np.clip(-mu, 0, None))

    fig, ax = plt.subplots(figsize=(6.8, 5.1), constrained_layout=True)
    plot_masked(ax, mu, np.zeros_like(mu), mu <= 0, color=STABLE, linewidth=2.6)
    plot_masked(ax, mu, np.zeros_like(mu), mu >= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    plot_masked(ax, mu, branch, mu <= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    plot_masked(ax, mu, -branch, mu <= 0, color=UNSTABLE, linewidth=2.6, linestyle="--")
    ax.scatter([0], [0], s=34, color="#111827", zorder=5)
    ax.set_title("Bifurcacion pitchfork subcritica")
    base_bif_axis(ax, (-2, 2), (-1.7, 1.7))
    bifurcation_legend(ax)
    save_figure(fig, "bif_g_pitchfork_sub")


def fig_periodica() -> None:
    mu = np.linspace(-2.0, 2.0, 700)
    ks = range(-2, 3)

    fig, ax = plt.subplots(figsize=(7.6, 5.1), constrained_layout=True)
    for k in ks:
        y_star = np.full_like(mu, k * np.pi)
        if k % 2 == 0:
            plot_masked(ax, mu, y_star, mu <= 0, color=STABLE, linewidth=2.3)
            plot_masked(ax, mu, y_star, mu >= 0, color=UNSTABLE, linewidth=2.3, linestyle="--")
        else:
            plot_masked(ax, mu, y_star, mu <= 0, color=UNSTABLE, linewidth=2.3, linestyle="--")
            plot_masked(ax, mu, y_star, mu >= 0, color=STABLE, linewidth=2.3)

    ax.axvline(0, color="#9ca3af", linewidth=1.0, linestyle=":")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.35 * np.pi, 2.35 * np.pi)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$y^*$")
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])
    ax.set_title("Bifurcacion periodica")
    style_axis(ax)
    bifurcation_legend(ax)
    save_figure(fig, "bif_i_periodica")


def plot_branch_by_stability(ax: plt.Axes, mu: np.ndarray, branch: np.ndarray) -> None:
    valid = ~np.isnan(branch)
    stable_mask = valid & ((3.0 * branch**2 - 3.0 * mu) < -1.0e-8)
    unstable_mask = valid & ((3.0 * branch**2 - 3.0 * mu) > 1.0e-8)
    plot_masked(ax, mu, branch, stable_mask, color=STABLE, linewidth=2.5)
    plot_masked(ax, mu, branch, unstable_mask, color=UNSTABLE, linewidth=2.5, linestyle="--")


def fig_cusp() -> None:
    mu = np.linspace(-0.5, 1.0, 900)
    branches = [np.full_like(mu, np.nan) for _ in range(3)]

    for i, mu_val in enumerate(mu):
        roots = np.roots([1.0, 0.0, -3.0 * mu_val, mu_val])
        real_roots = sorted(root.real for root in roots if abs(root.imag) < 1.0e-8)
        if len(real_roots) == 1:
            branches[1][i] = real_roots[0]
        else:
            for j, root in enumerate(real_roots):
                branches[j][i] = root

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    ax0, ax1 = axes

    for branch in branches:
        plot_branch_by_stability(ax0, mu, branch)
    ax0.scatter([0.0, 0.25], [0.0, 0.5], s=30, color="#111827", zorder=5)
    ax0.axvline(0.25, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax0.text(0.265, 0.7, r"$\mu=1/4$", color="#374151")
    ax0.set_title(r"Equilibrios de $y^3-3\mu y+\mu=0$")
    base_bif_axis(ax0, (-0.5, 1.0), (-2.1, 2.1))
    bifurcation_legend(ax0)

    r = np.linspace(0.0, 3.4, 500)
    h = 2.0 * (r / 3.0) ** 1.5
    ax1.plot(r, h, color=SOFT, linewidth=2.2)
    ax1.plot(r, -h, color=SOFT, linewidth=2.2)
    mu_line = np.linspace(-0.2, 1.0, 300)
    r_line = 3.0 * mu_line
    h_line = mu_line
    ax1.plot(r_line, h_line, color=ACCENT, linewidth=2.6, label=r"trayectoria $(r,h)=(3\mu,\mu)$")
    ax1.scatter([0.0, 0.75], [0.0, 0.25], s=30, color="#111827", zorder=5)
    ax1.set_xlim(-0.7, 3.2)
    ax1.set_ylim(-1.0, 1.1)
    ax1.set_xlabel(r"$r$")
    ax1.set_ylabel(r"$h$")
    ax1.set_title("Seccion uniparametrica de la cuspide")
    style_axis(ax1)
    ax1.legend(loc="upper left")

    save_figure(fig, "bif_j_cusp")


def main() -> None:
    fig_ej1()
    fig_ej2()
    fig_ej3_geometrico()
    fig_ej3_soluciones()
    fig_pitchfork_super("bif_a_pitchfork_sup", "Bifurcacion pitchfork supercritica")
    fig_transcritical("bif_c_transcritic", "Bifurcacion transcritica")
    fig_pitchfork_super("bif_d_pitchfork_sup2", "Bifurcacion pitchfork supercritica")
    fig_saddle_node()
    fig_pitchfork_sub()
    fig_transcritical("bif_h_transcritic2", "Bifurcacion transcritica")
    fig_periodica()
    fig_cusp()


if __name__ == "__main__":
    main()
