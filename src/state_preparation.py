"""
state_preparation.py
Discretises the log-normal distribution of S_T and builds
the quantum state preparation operator A for QAE.
"""

import numpy as np
import scipy.stats as stats
import pennylane as qml
from typing import Tuple


def discretise_lognormal(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_qubits: int, n_std: float = 3.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Discretise the log-normal distribution of S_T into 2^n bins.

    Parameters
    ----------
    n_qubits : number of state qubits (2^n bins)
    n_std    : number of std deviations to cover (default 3.5 → ~99.95%)

    Returns
    -------
    s_values     : stock price at each bin midpoint
    probs        : probability mass of each bin
    norm_payoffs : payoff normalised to [0, 1]
    max_payoff   : scaling factor to recover actual prices
    """
    n_bins    = 2 ** n_qubits
    mu_log    = np.log(S0) + (r - 0.5 * sigma ** 2) * T
    sigma_log = sigma * np.sqrt(T)

    S_min = np.exp(mu_log - n_std * sigma_log)
    S_max = np.exp(mu_log + n_std * sigma_log)

    edges    = np.linspace(S_min, S_max, n_bins + 1)
    s_values = 0.5 * (edges[:-1] + edges[1:])

    cdf_hi = stats.lognorm.cdf(edges[1:],  s=sigma_log, scale=np.exp(mu_log))
    cdf_lo = stats.lognorm.cdf(edges[:-1], s=sigma_log, scale=np.exp(mu_log))
    probs   = np.clip(cdf_hi - cdf_lo, 0, None)
    probs  /= probs.sum()

    raw_payoffs  = np.maximum(s_values - K, 0.0)
    max_payoff   = raw_payoffs.max()
    norm_payoffs = raw_payoffs / max_payoff if max_payoff > 0 else raw_payoffs

    return s_values, probs, norm_payoffs, max_payoff


def state_prep_operator(
    probs: np.ndarray,
    norm_payoffs: np.ndarray,
    state_wires: list,
    ancilla_wire: int,
) -> None:
    """
    PennyLane implementation of the A operator.

    Encodes:
      - probability distribution of S_T → state qubit amplitudes
      - normalised payoff → ancilla rotation angle

    Circuit effect:
      A|0⟩ = Σ_i sqrt(p_i) |i⟩ ⊗ (sqrt(f_i)|1⟩ + sqrt(1-f_i)|0⟩)

    where p_i = probability of bin i, f_i = normalised payoff of bin i.
    """
    # Step 1: load sqrt(probs) as amplitudes on state qubits
    amplitudes = np.sqrt(probs)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    qml.MottonenStatePreparation(amplitudes, wires=state_wires)

    # Step 2: rotate ancilla by expected payoff angle
    # E[norm_payoff] under this distribution
    expected_f = float(np.dot(probs, norm_payoffs))
    angle = 2.0 * np.arcsin(np.sqrt(np.clip(expected_f, 0.0, 1.0)))
    qml.RY(angle, wires=ancilla_wire)


def expected_payoff_exact(
    probs: np.ndarray, norm_payoffs: np.ndarray, max_payoff: float,
    r: float, T: float
) -> float:
    """
    Compute the exact option price implied by the discretised distribution.
    This is the 'oracle' value that QAE aims to estimate.
    """
    a_exact = float(np.dot(probs, norm_payoffs))
    return np.exp(-r * T) * a_exact * max_payoff


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0

    print("Discretisation quality vs qubit count:\n")
    print(f"{'Qubits':>8}  {'Bins':>6}  {'Price':>10}  {'Disc Error':>12}")
    print("-" * 45)
    for n in [3, 4, 5, 6, 7]:
        s_v, prb, npf, mpf = discretise_lognormal(S0, K, r, sigma, T, n)
        price = expected_payoff_exact(prb, npf, mpf, r, T)
        from black_scholes import call_price
        err = abs(price - call_price(S0, K, r, sigma, T))
        print(f"{n:>8}  {2**n:>6}  ${price:>9.4f}  ${err:>11.6f}")
