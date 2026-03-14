"""
black_scholes.py
Closed-form Black-Scholes European option pricer.
Used as ground-truth benchmark throughout the project.
"""

import numpy as np
import scipy.stats as stats


def call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes closed-form price for a European call option.

    Parameters
    ----------
    S0    : float  Current stock price
    K     : float  Strike price
    r     : float  Risk-free rate (annual)
    sigma : float  Volatility (annual)
    T     : float  Time to maturity (years)

    Returns
    -------
    float  Option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def put_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-Scholes closed-form price for a European put option (via put-call parity).
    """
    call = call_price(S0, K, r, sigma, T)
    return call - S0 + K * np.exp(-r * T)


def greeks(S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    """
    Returns the main Black-Scholes Greeks for a European call.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = stats.norm.cdf(d1)
    gamma = stats.norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega  = S0 * stats.norm.pdf(d1) * np.sqrt(T)
    theta = (- (S0 * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.20, 1.0
    price = call_price(S0, K, r, sigma, T)
    print(f"European call price (Black-Scholes): ${price:.4f}")
    print(f"Greeks: {greeks(S0, K, r, sigma, T)}")
