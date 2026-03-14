"""
plotting.py
All visualisation utilities for the QAE options pricing project.
Each function saves a figure to results/figures/ and returns the fig object.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from typing import List, Optional

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────
COLOR_CLASSICAL = "#1D9E75"   # teal
COLOR_QUANTUM   = "#7F77DD"   # purple
COLOR_NOISE     = "#D85A30"   # coral
COLOR_REF       = "#888780"   # gray

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
})


def plot_discretisation(s_values, probs, norm_payoffs, K, n_qubits, save=True):
    """Fig 1 — discretised distribution and payoff."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(s_values, probs, width=(s_values[1] - s_values[0]) * 0.9,
                color=COLOR_QUANTUM, alpha=0.8)
    axes[0].axvline(K, color=COLOR_NOISE, linestyle="--", lw=1.5,
                    label=f"Strike K=${K}")
    axes[0].set_title(f"Discretised log-normal distribution of $S_T$\n({2**n_qubits} bins, {n_qubits} qubits)")
    axes[0].set_xlabel("Stock price $S_T$")
    axes[0].set_ylabel("Probability mass")
    axes[0].legend()

    raw_payoffs = norm_payoffs * norm_payoffs.max() if norm_payoffs.max() > 0 else norm_payoffs
    axes[1].bar(s_values, np.maximum(s_values - K, 0),
                width=(s_values[1] - s_values[0]) * 0.9,
                color=COLOR_CLASSICAL, alpha=0.8)
    axes[1].axvline(K, color=COLOR_NOISE, linestyle="--", lw=1.5,
                    label=f"Strike K=${K}")
    axes[1].set_title("Payoff function max$(S_T - K,\\ 0)$")
    axes[1].set_xlabel("Stock price $S_T$")
    axes[1].set_ylabel("Payoff ($)")
    axes[1].legend()

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig1_discretisation.png", bbox_inches="tight")
    return fig


def plot_convergence_comparison(
    N_classical: List[int], cmc_errors: List[float],
    N_quantum:   List[int], qae_errors: List[float],
    slope_c: float, slope_q: float,
    save=True,
):
    """Fig 2 — the key result: log-log convergence + speedup ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: log-log convergence ──────────────────────────────────────────
    ax = axes[0]
    ax.loglog(N_classical, cmc_errors, "o-", color=COLOR_CLASSICAL,
              lw=2, ms=7, label="Classical Monte Carlo")
    ax.loglog(N_quantum, qae_errors, "s-", color=COLOR_QUANTUM,
              lw=2, ms=7, label="Quantum AE")

    # Reference slope lines
    N_ref = np.array([min(N_classical), max(N_classical)], dtype=float)
    c_c = cmc_errors[0] * np.sqrt(N_classical[0])
    c_q = qae_errors[0] * N_quantum[0]
    ax.loglog(N_ref, c_c / np.sqrt(N_ref), "--", color=COLOR_CLASSICAL,
              alpha=0.35, lw=1.5, label="$O(1/\\sqrt{N})$ ref")
    ax.loglog(N_ref, c_q / N_ref, "--", color=COLOR_QUANTUM,
              alpha=0.35, lw=1.5, label="$O(1/N)$ ref")

    mid_c = len(N_classical) // 2
    mid_q = len(N_quantum)   // 2
    ax.annotate(f"slope ≈ {slope_c:.2f}",
                xy=(N_classical[mid_c], cmc_errors[mid_c]),
                xytext=(N_classical[mid_c] * 0.3, cmc_errors[mid_c] * 3),
                fontsize=9, color=COLOR_CLASSICAL,
                arrowprops=dict(arrowstyle="->", color=COLOR_CLASSICAL, lw=1))
    ax.annotate(f"slope ≈ {slope_q:.2f}",
                xy=(N_quantum[mid_q], qae_errors[mid_q]),
                xytext=(N_quantum[mid_q] * 0.3, qae_errors[mid_q] * 0.25),
                fontsize=9, color=COLOR_QUANTUM,
                arrowprops=dict(arrowstyle="->", color=COLOR_QUANTUM, lw=1))

    ax.set_xlabel("Number of evaluations $N$")
    ax.set_ylabel("Absolute pricing error ($)")
    ax.set_title("Convergence comparison (log-log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Right: speedup ratio ───────────────────────────────────────────────
    ax2 = axes[1]
    cmc_interp = np.interp(N_quantum, N_classical, cmc_errors)
    ratio = cmc_interp / (np.array(qae_errors) + 1e-12)

    bars = ax2.bar(range(len(N_quantum)), ratio,
                   color=COLOR_QUANTUM, alpha=0.75, width=0.6)
    ax2.axhline(1.0, color=COLOR_REF, linestyle="--", lw=1.2,
                label="Break-even (ratio = 1)")
    for bar, r_val in zip(bars, ratio):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{r_val:.1f}x", ha="center", va="bottom", fontsize=9,
                 color=COLOR_QUANTUM)

    ax2.set_xticks(range(len(N_quantum)))
    ax2.set_xticklabels([f"{n}" for n in N_quantum], rotation=30)
    ax2.set_xlabel("Number of evaluations $N$")
    ax2.set_ylabel("Speedup ratio\n(Classical error / Quantum error)")
    ax2.set_title("Quantum speedup factor\n(higher = quantum is more accurate)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, axis="y")

    plt.suptitle("QAE vs Classical Monte Carlo — Options Pricing Convergence",
                 fontsize=13, fontweight="500", y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig2_convergence_comparison.png", bbox_inches="tight")
    return fig


def plot_noise_analysis(
    noise_levels: List[float],
    noise_errors: List[float],
    noise_stds: List[float],
    classical_reference_error: float,
    crossover_noise: Optional[float],
    n_shots: int,
    save=True,
):
    """Fig 3 — effect of gate noise on QAE accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(noise_levels, noise_errors, yerr=noise_stds,
                fmt="o-", color=COLOR_NOISE, lw=2, ms=7, capsize=4,
                label="QAE pricing error")
    ax.axhline(classical_reference_error, color=COLOR_CLASSICAL,
               linestyle="--", lw=1.5,
               label=f"Classical MC error (N={n_shots})")

    if crossover_noise is not None:
        ax.axvline(crossover_noise, color=COLOR_REF, linestyle=":", lw=1.2,
                   label=f"Crossover ≈ {crossover_noise*100:.1f}%")
        ax.fill_betweenx(
            [0, max(max(noise_errors), classical_reference_error) * 1.2],
            0, crossover_noise, alpha=0.06, color=COLOR_QUANTUM,
            label="Quantum advantage region"
        )

    ax.set_xlabel("Depolarising noise rate per gate")
    ax.set_ylabel("Absolute pricing error ($)")
    ax.set_title(f"Effect of hardware noise on QAE accuracy\n(N = {n_shots} shots)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig3_noise_analysis.png", bbox_inches="tight")
    return fig


def plot_qubit_ablation(qubit_counts: List[int], disc_errors: List[float], save=True):
    """Fig 4 — discretisation error vs number of state qubits."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(qubit_counts, disc_errors, "o-", color=COLOR_QUANTUM, lw=2, ms=8)
    ax.set_xlabel("Number of state qubits")
    ax.set_ylabel("Discretisation error ($ — log scale)")
    ax.set_title("Discretisation error vs qubit count\n(more qubits → better approximation, deeper circuit)")
    ax.set_xticks(qubit_counts)
    ax.set_xticklabels([f"{n}\n({2**n} bins)" for n in qubit_counts])
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig4_qubit_ablation.png", bbox_inches="tight")
    return fig
