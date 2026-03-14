"""
qae_circuit.py
Quantum Amplitude Estimation circuit and convergence study.
Uses PennyLane's default.qubit simulator.

The core insight: QAE estimates E[payoff] with O(1/N) error
vs classical Monte Carlo's O(1/sqrt(N)) — a quadratic speedup.
"""

import numpy as np
import pennylane as qml
from typing import List, Tuple
from state_preparation import state_prep_operator


def build_qae_device(n_state_qubits: int, shots: int | None = None) -> qml.Device:
    """Return a PennyLane device with n_state_qubits + 1 ancilla."""
    return qml.device("default.qubit", wires=n_state_qubits + 1, shots=shots)


def qae_exact_amplitude(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
    dev: qml.Device,
) -> float:
    """
    Exact QAE: compute P(ancilla=|1⟩) analytically (no shot noise).
    Returns the amplitude a = E[norm_payoff].
    """
    @qml.qnode(dev)
    def circuit():
        state_prep_operator(probs, norm_payoffs, state_wires, ancilla_wire)
        return qml.expval(qml.PauliZ(wires=ancilla_wire))

    z_exp = float(circuit())
    # P(|1⟩) = (1 - <Z>) / 2
    return (1.0 - z_exp) / 2.0


def qae_shot_estimate(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
    n_shots: int,
    dev: qml.Device,
) -> float:
    """
    Shot-based QAE: measure the ancilla qubit n_shots times.
    Returns estimated amplitude a ≈ E[norm_payoff].
    This simulates the shot noise of a real quantum device.
    """
    @qml.qnode(dev, shots=n_shots)
    def circuit():
        state_prep_operator(probs, norm_payoffs, state_wires, ancilla_wire)
        return qml.sample(wires=ancilla_wire)

    samples = circuit()
    return float(np.mean(samples))


def amplitude_to_price(
    a: float, max_payoff: float, r: float, T: float
) -> float:
    """Convert estimated amplitude back to option price."""
    return np.exp(-r * T) * a * max_payoff


def convergence_study(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
    max_payoff: float,
    r: float,
    T: float,
    N_values: List[int],
    n_repeats: int = 30,
    reference_price: float | None = None,
) -> Tuple[List[float], List[float]]:
    """
    Run QAE at each N (shot count) and record mean price and std.

    Parameters
    ----------
    N_values    : list of shot counts to evaluate
    n_repeats   : independent runs per N
    reference_price : if given, errors are computed vs this price

    Returns
    -------
    means : mean price estimate at each N
    stds  : std of estimates across repeats
    """
    means, stds = [], []
    for N in N_values:
        dev = build_qae_device(len(state_wires), shots=N)
        estimates = []
        for _ in range(n_repeats):
            a_est = qae_shot_estimate(
                probs, norm_payoffs, state_wires, ancilla_wire, N, dev
            )
            price = amplitude_to_price(a_est, max_payoff, r, T)
            estimates.append(price)
        means.append(float(np.mean(estimates)))
        stds.append(float(np.std(estimates)))
    return means, stds


if __name__ == "__main__":
    from state_preparation import discretise_lognormal, expected_payoff_exact
    from black_scholes import call_price

    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
    n_qubits     = 4
    state_wires  = list(range(n_qubits))
    ancilla_wire = n_qubits

    s_vals, probs, norm_payoffs, max_payoff = discretise_lognormal(
        S0, K, r, sigma, T, n_qubits
    )

    bs_price   = call_price(S0, K, r, sigma, T)
    disc_price = expected_payoff_exact(probs, norm_payoffs, max_payoff, r, T)
    dev_exact  = build_qae_device(n_qubits)
    a_exact    = qae_exact_amplitude(probs, norm_payoffs, state_wires, ancilla_wire, dev_exact)
    qae_price  = amplitude_to_price(a_exact, max_payoff, r, T)

    print(f"Black-Scholes truth:  ${bs_price:.4f}")
    print(f"Discretised exact:    ${disc_price:.4f}")
    print(f"QAE circuit exact:    ${qae_price:.4f}")
    print(f"Discretisation gap:   ${abs(qae_price - bs_price):.5f}")

    N_values = [100, 500, 1000, 2000, 5000]
    print(f"\nConvergence study (n_repeats=20):\n")
    print(f"{'N':>6}  {'Mean Price':>12}  {'Std':>10}  {'Abs Error':>12}")
    print("-" * 48)
    means, stds = convergence_study(
        probs, norm_payoffs, state_wires, ancilla_wire,
        max_payoff, r, T, N_values, n_repeats=20,
        reference_price=qae_price
    )
    for N, m, s in zip(N_values, means, stds):
        print(f"{N:>6}  ${m:>11.4f}  ${s:>9.5f}  ${abs(m - qae_price):>11.6f}")
