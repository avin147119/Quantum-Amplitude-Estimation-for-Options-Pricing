"""
classical_mc.py
Classical Monte Carlo option pricer and convergence study.
Establishes the O(1/sqrt(N)) baseline that QAE is benchmarked against.
"""

import numpy as np
from typing import Tuple, List


def price_european_call(
    S0: float, K: float, r: float, sigma: float, T: float,
    N: int, seed: int | None = None
) -> float:
    """
    Price a European call option using Monte Carlo simulation.

    Parameters
    ----------
    S0, K, r, sigma, T : option parameters
    N    : number of simulated paths
    seed : random seed (optional)

    Returns
    -------
    float  Estimated option price
    """
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(N)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0.0)
    return np.exp(-r * T) * np.mean(payoffs)


def convergence_study(
    S0: float, K: float, r: float, sigma: float, T: float,
    N_values: List[int],
    n_repeats: int = 50,
    base_seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """
    Run repeated MC trials at each N to measure mean error and std.

    Parameters
    ----------
    N_values  : list of evaluation counts to test
    n_repeats : number of independent runs per N (for stable error estimate)
    base_seed : base random seed

    Returns
    -------
    means : list of mean price estimates per N
    stds  : list of std across repeats per N
    """
    means, stds = [], []
    for i, N in enumerate(N_values):
        estimates = [
            price_european_call(S0, K, r, sigma, T, N, seed=base_seed + i * n_repeats + j)
            for j in range(n_repeats)
        ]
        means.append(float(np.mean(estimates)))
        stds.append(float(np.std(estimates)))
    return means, stds


def theoretical_std(sigma: float, S0: float, r: float, T: float, N: int) -> float:
    """
    Theoretical standard error of MC estimator: std(payoff) / sqrt(N).
    Provides a sanity check against empirical results.
    """
    n_samples = 100_000
    rng = np.random.default_rng(0)
    Z   = rng.standard_normal(n_samples)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff_std = np.std(np.maximum(S_T - 105.0, 0)) * np.exp(-r * T)
    return payoff_std / np.sqrt(N)


if __name__ == "__main__":
    from black_scholes import call_price

    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
    bs = call_price(S0, K, r, sigma, T)

    N_values = [100, 500, 1000, 5000, 10000]
    means, stds = convergence_study(S0, K, r, sigma, T, N_values, n_repeats=50)

    print(f"Black-Scholes truth: ${bs:.4f}\n")
    print(f"{'N':>8}  {'Estimate':>10}  {'Abs Error':>10}  {'Std':>10}")
    print("-" * 45)
    for N, m, s in zip(N_values, means, stds):
        print(f"{N:>8}  ${m:>9.4f}  ${abs(m-bs):>9.5f}  ${s:>9.5f}")
