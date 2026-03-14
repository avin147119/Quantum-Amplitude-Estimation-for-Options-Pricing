"""
noise_analysis.py
Simulates the effect of hardware noise on QAE using PennyLane's
default.mixed device with depolarising channels.

Key question: at what gate error rate does QAE lose its advantage
over classical Monte Carlo? This is the honest NISQ analysis.
"""

import numpy as np
import pennylane as qml
from typing import List, Tuple


def qae_with_depolarising_noise(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
    max_payoff: float,
    r: float,
    T: float,
    n_shots: int,
    noise_prob: float,
) -> float:
    """
    Run QAE with depolarising noise on all qubits after state preparation.

    Parameters
    ----------
    noise_prob : per-qubit depolarising error probability (0.0 = noiseless)

    Returns
    -------
    float  Estimated option price under noisy conditions
    """
    n_wires = len(state_wires) + 1
    dev = qml.device("default.mixed", wires=n_wires, shots=n_shots)

    @qml.qnode(dev)
    def noisy_circuit():
        # State preparation
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        qml.MottonenStatePreparation(amplitudes, wires=state_wires)
        # Depolarising noise on state qubits
        for w in state_wires:
            qml.DepolarizingChannel(noise_prob, wires=w)

        # Ancilla rotation
        expected_f = float(np.dot(probs, norm_payoffs))
        angle = 2.0 * np.arcsin(np.sqrt(np.clip(expected_f, 0.0, 1.0)))
        qml.RY(angle, wires=ancilla_wire)
        qml.DepolarizingChannel(noise_prob, wires=ancilla_wire)

        return qml.sample(wires=ancilla_wire)

    samples = noisy_circuit()
    a_est   = float(np.mean(samples))
    return np.exp(-r * T) * a_est * max_payoff


def noise_sweep(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
    max_payoff: float,
    r: float,
    T: float,
    noise_levels: List[float],
    n_shots: int = 1000,
    n_repeats: int = 20,
) -> Tuple[List[float], List[float]]:
    """
    Sweep over noise levels and measure QAE accuracy at each.

    Returns
    -------
    means : mean price estimate at each noise level
    stds  : std across repeats at each noise level
    """
    means, stds = [], []
    for p_noise in noise_levels:
        estimates = [
            qae_with_depolarising_noise(
                probs, norm_payoffs, state_wires, ancilla_wire,
                max_payoff, r, T, n_shots, p_noise
            )
            for _ in range(n_repeats)
        ]
        means.append(float(np.mean(estimates)))
        stds.append(float(np.std(estimates)))
    return means, stds


def find_crossover_noise(
    qae_errors_by_noise: List[float],
    noise_levels: List[float],
    classical_reference_error: float,
) -> float | None:
    """
    Find the noise level at which QAE error exceeds the classical MC error.
    Returns the estimated crossover noise probability (or None if never crossed).
    """
    for i, (noise, qae_err) in enumerate(zip(noise_levels, qae_errors_by_noise)):
        if qae_err >= classical_reference_error:
            if i == 0:
                return noise_levels[0]
            # Linear interpolation between i-1 and i
            x0, x1 = noise_levels[i - 1], noise_levels[i]
            y0, y1 = qae_errors_by_noise[i - 1], qae_errors_by_noise[i]
            t = (classical_reference_error - y0) / (y1 - y0 + 1e-12)
            return x0 + t * (x1 - x0)
    return None


if __name__ == "__main__":
    from state_preparation import discretise_lognormal
    from black_scholes import call_price

    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
    n_qubits     = 4
    state_wires  = list(range(n_qubits))
    ancilla_wire = n_qubits

    s_vals, probs, norm_payoffs, max_payoff = discretise_lognormal(
        S0, K, r, sigma, T, n_qubits
    )
    bs_price = call_price(S0, K, r, sigma, T)

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    means, stds  = noise_sweep(
        probs, norm_payoffs, state_wires, ancilla_wire,
        max_payoff, r, T, noise_levels, n_shots=1000, n_repeats=20
    )

    print(f"Black-Scholes truth: ${bs_price:.4f}\n")
    print(f"{'Noise':>8}  {'Mean Price':>12}  {'Abs Error':>12}")
    print("-" * 38)
    errors = []
    for noise, mean in zip(noise_levels, means):
        err = abs(mean - bs_price)
        errors.append(err)
        print(f"{noise:>8.3f}  ${mean:>11.4f}  ${err:>11.5f}")

    # Classical MC reference at same N=1000
    from classical_mc import price_european_call
    cmc_ref = abs(price_european_call(S0, K, r, sigma, T, N=1000) - bs_price)
    crossover = find_crossover_noise(errors, noise_levels, cmc_ref)
    print(f"\nClassical MC error at N=1000: ${cmc_ref:.5f}")
    if crossover:
        print(f"QAE advantage lost at noise ≈ {crossover:.3f} ({crossover*100:.1f}%)")
    else:
        print("QAE advantage holds across all tested noise levels.")
