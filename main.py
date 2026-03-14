"""
main.py
Run the full QAE options pricing experiment from the command line.

Usage:
    python main.py                  # Run with default parameters
    python main.py --qubits 5       # Use 5 state qubits
    python main.py --repeats 50     # More repeats for stable statistics
    python main.py --no-noise       # Skip noise analysis (faster)

All figures are saved to results/figures/.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from black_scholes     import call_price
from classical_mc      import convergence_study as cmc_convergence_study
from state_preparation import discretise_lognormal, expected_payoff_exact
from qae_circuit       import (
    build_qae_device, qae_exact_amplitude,
    amplitude_to_price, convergence_study as qae_convergence_study,
)
from noise_analysis    import noise_sweep, find_crossover_noise
from plotting          import (
    plot_discretisation, plot_convergence_comparison,
    plot_noise_analysis, plot_qubit_ablation,
)


# ── Default option parameters ──────────────────────────────────────────────
DEFAULT_PARAMS = dict(S0=100.0, K=105.0, r=0.05, sigma=0.20, T=1.0)


def parse_args():
    p = argparse.ArgumentParser(description="QAE Options Pricing Experiment")
    p.add_argument("--qubits",   type=int,   default=4,    help="State qubits (default: 4)")
    p.add_argument("--repeats",  type=int,   default=30,   help="Repeats per N (default: 30)")
    p.add_argument("--no-noise", action="store_true",      help="Skip noise analysis")
    p.add_argument("--no-plots", action="store_true",      help="Skip saving figures")
    return p.parse_args()


def divider(title=""):
    width = 58
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, width - len(title) - 5)}")
    else:
        print("─" * width)


def main():
    args   = parse_args()
    params = DEFAULT_PARAMS
    save   = not args.no_plots

    print("\n" + "═" * 58)
    print("  Quantum Amplitude Estimation for Options Pricing")
    print("  Convergence Advantage Demonstration")
    print("═" * 58)
    print(f"  S0={params['S0']}, K={params['K']}, r={params['r']}, "
          f"σ={params['sigma']}, T={params['T']}")
    print(f"  State qubits: {args.qubits}  |  Repeats per N: {args.repeats}")

    # ── 1. Ground truth ────────────────────────────────────────────────────
    divider("1. Black-Scholes Ground Truth")
    bs_price = call_price(**params)
    print(f"  European call price: ${bs_price:.4f}")

    # ── 2. Discretisation ──────────────────────────────────────────────────
    divider("2. Distribution Discretisation")
    s_vals, probs, norm_payoffs, max_payoff = discretise_lognormal(
        **params, n_qubits=args.qubits
    )
    disc_price = expected_payoff_exact(probs, norm_payoffs, max_payoff,
                                        params["r"], params["T"])
    print(f"  Bins: {2**args.qubits}  |  Disc. price: ${disc_price:.4f}  "
          f"|  Disc. error: ${abs(disc_price - bs_price):.5f}")

    if save:
        plot_discretisation(s_vals, probs, norm_payoffs,
                            params["K"], args.qubits)
        print("  Saved: fig1_discretisation.png")

    # ── 3. QAE exact amplitude ─────────────────────────────────────────────
    divider("3. QAE Circuit (Exact)")
    state_wires  = list(range(args.qubits))
    ancilla_wire = args.qubits
    dev_exact    = build_qae_device(args.qubits)
    a_exact      = qae_exact_amplitude(probs, norm_payoffs,
                                        state_wires, ancilla_wire, dev_exact)
    qae_exact_price = amplitude_to_price(a_exact, max_payoff,
                                          params["r"], params["T"])
    print(f"  Exact QAE price: ${qae_exact_price:.4f}")

    # ── 4. Classical MC convergence ────────────────────────────────────────
    divider("4. Classical Monte Carlo Convergence Study")
    N_classical = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
    print(f"  N values: {N_classical}")
    cmc_means, cmc_stds = cmc_convergence_study(
        **params, N_values=N_classical, n_repeats=args.repeats
    )
    cmc_errors = [abs(m - bs_price) for m in cmc_means]
    slope_c = float(np.polyfit(np.log(N_classical), np.log(np.array(cmc_errors) + 1e-12), 1)[0])
    print(f"  Empirical convergence slope: {slope_c:.3f}  (theory: -0.500)")

    # ── 5. QAE convergence ─────────────────────────────────────────────────
    divider("5. Quantum AE Convergence Study")
    N_quantum = [50, 100, 250, 500, 1000, 2000, 5000]
    print(f"  N values: {N_quantum}")
    qae_means, qae_stds = qae_convergence_study(
        probs, norm_payoffs, state_wires, ancilla_wire,
        max_payoff, params["r"], params["T"],
        N_quantum, n_repeats=args.repeats,
        reference_price=qae_exact_price,
    )
    qae_errors = [abs(m - qae_exact_price) for m in qae_means]
    slope_q = float(np.polyfit(np.log(N_quantum), np.log(np.array(qae_errors) + 1e-12), 1)[0])
    print(f"  Empirical convergence slope: {slope_q:.3f}  (theory: -1.000)")

    speedup = np.interp(1000, N_classical, cmc_errors) / (
        np.interp(1000, N_quantum, qae_errors) + 1e-12
    )
    print(f"  Quantum speedup at N=1000:   {speedup:.1f}x")

    if save:
        plot_convergence_comparison(
            N_classical, cmc_errors, N_quantum, qae_errors, slope_c, slope_q
        )
        print("  Saved: fig2_convergence_comparison.png")

    # ── 6. Noise analysis ──────────────────────────────────────────────────
    if not args.no_noise:
        divider("6. NISQ Noise Analysis")
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
        n_shots_noise = 1000
        print(f"  Sweeping noise levels: {noise_levels}")
        noise_means, noise_stds = noise_sweep(
            probs, norm_payoffs, state_wires, ancilla_wire,
            max_payoff, params["r"], params["T"],
            noise_levels, n_shots=n_shots_noise, n_repeats=20,
        )
        noise_errors   = [abs(m - bs_price) for m in noise_means]
        cmc_ref_error  = float(np.interp(n_shots_noise, N_classical, cmc_errors))
        crossover      = find_crossover_noise(noise_errors, noise_levels, cmc_ref_error)

        if crossover:
            print(f"  Classical MC error at N={n_shots_noise}: ${cmc_ref_error:.5f}")
            print(f"  Quantum advantage lost at noise ≈ {crossover*100:.1f}%")
        else:
            print("  Quantum advantage holds across all tested noise levels.")

        if save:
            plot_noise_analysis(noise_levels, noise_errors, noise_stds,
                                cmc_ref_error, crossover, n_shots_noise)
            print("  Saved: fig3_noise_analysis.png")

    # ── 7. Qubit ablation ──────────────────────────────────────────────────
    divider("7. Qubit Ablation Study")
    qubit_counts = [3, 4, 5, 6]
    disc_errors  = []
    for nq in qubit_counts:
        sv, pb, npf, mpf = discretise_lognormal(**params, n_qubits=nq)
        dp = expected_payoff_exact(pb, npf, mpf, params["r"], params["T"])
        disc_errors.append(abs(dp - bs_price))
        print(f"  n={nq} ({2**nq:>3d} bins): disc_error=${disc_errors[-1]:.6f}")

    if save:
        plot_qubit_ablation(qubit_counts, disc_errors)
        print("  Saved: fig4_qubit_ablation.png")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  EXPERIMENT COMPLETE — RESULTS SUMMARY")
    print("═" * 58)
    print(f"  Black-Scholes truth:          ${bs_price:.4f}")
    print(f"  QAE circuit exact value:      ${qae_exact_price:.4f}")
    print()
    print(f"  Classical MC slope:  {slope_c:+.3f}  (theory: -0.500)")
    print(f"  Quantum AE slope:    {slope_q:+.3f}  (theory: -1.000)")
    print()
    print(f"  Quantum speedup at N=1000:    {speedup:.1f}x")
    if not args.no_noise and crossover:
        print(f"  Noise crossover:              ~{crossover*100:.1f}% per gate")
    if save:
        print(f"\n  All figures saved to:  results/figures/")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()
