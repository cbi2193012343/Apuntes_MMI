import numpy as np
import matplotlib.pyplot as plt


def save_fig_logistico_cosecha(path: str) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    lambdas = [0.10, 0.25, 0.40]
    titles = [r"$\lambda<1/4$", r"$\lambda=1/4$", r"$\lambda>1/4$"]

    for ax, lam, title in zip(axs[0], lambdas, titles):
        u = np.linspace(-0.3, 1.3, 600)
        f = u * (1 - u) - lam
        ax.axhline(0, color="black", lw=1)
        ax.plot(u, f, color="#c03a2b", lw=2)
        ax.set_title(title)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.4, 0.45)
        ax.set_xlabel("u")
        ax.set_ylabel(r"$u(1-u)-\lambda$")

    lam = 0.10
    ustar = np.array([0.5 - np.sqrt(0.25 - lam), 0.5 + np.sqrt(0.25 - lam)])
    ax = axs[1, 0]
    ax.set_title("Linea de fase")
    ax.axhline(0, color="black", lw=1)
    ax.scatter(ustar, [0, 0], color="black", zorder=3)
    ax.text(ustar[0], 0.05, r"$u_-$", ha="center")
    ax.text(ustar[1], 0.05, r"$u_+$", ha="center")
    ax.text(0.05, -0.15, "estable", fontsize=10)
    ax.text(0.92, -0.15, "estable", fontsize=10)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 0.3)
    ax.axis("off")

    ax = axs[1, 1]
    lamv = np.linspace(0, 0.249, 400)
    u1 = 0.5 - np.sqrt(0.25 - lamv)
    u2 = 0.5 + np.sqrt(0.25 - lamv)
    ax.plot(lamv, u1, color="#2980b9", lw=2)
    ax.plot(lamv, u2, color="#2980b9", lw=2)
    ax.axvline(0.25, color="red", ls="--", lw=1)
    ax.scatter([0.25], [0.5], color="red", zorder=3)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$u^*$")
    ax.set_title("Bifurcacion silla-nodo")
    ax.set_xlim(-0.01, 0.4)
    ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Modelo logistico con cosecha")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fig_lotka_volterra(path: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    ax = axs[0]
    P = np.linspace(0, 5, 25)
    D = np.linspace(0, 5, 25)
    PP, DD = np.meshgrid(P, D)
    dP = PP * (1 - PP / 4) - 0.6 * PP * DD
    dD = -DD + 0.5 * PP * DD
    speed = np.hypot(dP, dD) + 1e-9
    ax.quiver(PP, DD, dP / speed, dD / speed, color="#1f77b4", alpha=0.75)
    pnull = np.linspace(0, 5, 400)
    ax.plot(pnull, np.zeros_like(pnull), color="black", lw=1)
    ax.plot(np.full_like(pnull, 2.0), pnull, color="red", lw=2, ls="--")
    ax.plot(pnull, 2 / np.maximum(pnull, 1e-3), color="green", lw=2)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Presa P")
    ax.set_ylabel("Depredador D")
    ax.set_title("Campo de fase y nullclines")

    ax = axs[1]
    x = np.linspace(0, 5, 400)
    ax.plot(x, x, color="#c0392b", lw=2, label=r"$f(P)=P$")
    ax.plot(x, x / (1 + x), color="#27ae60", lw=2, label=r"$f(P)=\frac{P}{1+P}$")
    ax.plot(x, x**2 / (1 + x**2), color="#2980b9", lw=2, label=r"$f(P)=\frac{P^2}{1+P^2}$")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("P")
    ax.set_ylabel("Respuesta funcional")
    ax.set_title("Respuestas funcionales")
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Dinamica de dos poblaciones")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fig_linealizacion(path: str) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    configs = [
        ("Fuente", np.array([[1.0, 0.2], [0.2, 1.0]]), "out"),
        ("Sumidero", np.array([[-1.0, -0.2], [-0.2, -1.0]]), "in"),
        ("Silla", np.array([[1.0, 0.0], [0.0, -1.0]]), "saddle"),
        ("Espiral", np.array([[0.2, -1.0], [1.0, 0.2]]), "spiral"),
    ]
    grid = np.linspace(-2, 2, 9)
    for ax, (title, A, kind) in zip(axs.flat, configs):
        X, Y = np.meshgrid(grid, grid)
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        S = np.hypot(U, V) + 1e-9
        ax.quiver(X, Y, U / S, V / S, color="#34495e", alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal", "box")
        ax.set_title(title)
    fig.suptitle("Clasificacion local en el plano")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fig_allee(path: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Phase portrait sketches for lambda < 1 and lambda > 1
    cases = [
        (axs[0], 0.6, "A < K"),
        (axs[1], 1.4, "A > K"),
    ]
    for ax, lam, title in cases:
        u = np.linspace(0, 1.8, 600)
        f = u * (1 - u) * (u / lam - 1)
        ax.axhline(0, color="black", lw=1)
        ax.plot(u, f, color="#8e44ad", lw=2)
        eqs = [0, lam, 1]
        eqs = sorted(set([x for x in eqs if 0 <= x <= 1.8]))
        ax.scatter(eqs, [0] * len(eqs), color="black", zorder=3)
        for x in eqs:
            ax.text(x, 0.03, f"{x:.1f}" if x not in (0, 1) else ("0" if x == 0 else "1"),
                    ha="center", fontsize=9)
        ax.set_xlim(0, 1.8)
        ax.set_ylim(-0.35, 0.35)
        ax.set_title(title)
        ax.set_xlabel("u")
        ax.set_ylabel(r"$u(1-u)(u/\lambda-1)$")

    fig.suptitle("Modelo logístico con efecto Allee")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    save_fig_logistico_cosecha("fig_14_logistico_cosecha.png")
    save_fig_lotka_volterra("fig_15_lotka_volterra.png")
    save_fig_linealizacion("fig_16_dinamica_local.png")
    save_fig_allee("fig_17_allee.png")
