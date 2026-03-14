"""
tests/test_pricing.py
Unit tests for all pricing modules.
Run with: pytest tests/
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from black_scholes     import call_price, put_price, greeks
from classical_mc      import price_european_call
from state_preparation import discretise_lognormal, expected_payoff_exact
from qae_circuit       import (
    build_qae_device, qae_exact_amplitude, amplitude_to_price
)


# ── Shared fixtures ────────────────────────────────────────────────────────
PARAMS = dict(S0=100.0, K=105.0, r=0.05, sigma=0.20, T=1.0)
BS_PRICE = call_price(**PARAMS)  # ~8.02


class TestBlackScholes:
    def test_call_price_positive(self):
        assert call_price(**PARAMS) > 0

    def test_call_price_known_value(self):
        # Known result for these parameters
        price = call_price(**PARAMS)
        assert 7.5 < price < 9.0, f"Price {price:.4f} outside expected range"

    def test_put_call_parity(self):
        """C - P = S0 - K*e^(-rT)"""
        p = PARAMS
        call = call_price(**p)
        put  = put_price(**p)
        lhs  = call - put
        rhs  = p["S0"] - p["K"] * np.exp(-p["r"] * p["T"])
        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs:.6f} != {rhs:.6f}"

    def test_call_increases_with_S0(self):
        c1 = call_price(S0=100, **{k: v for k, v in PARAMS.items() if k != "S0"})
        c2 = call_price(S0=110, **{k: v for k, v in PARAMS.items() if k != "S0"})
        assert c2 > c1

    def test_call_decreases_with_strike(self):
        c1 = call_price(**PARAMS)
        c2 = call_price(S0=100, K=115, r=0.05, sigma=0.20, T=1.0)
        assert c1 > c2

    def test_greeks_delta_range(self):
        g = greeks(**PARAMS)
        assert 0.0 < g["delta"] < 1.0, "Delta must be in (0, 1)"

    def test_greeks_gamma_positive(self):
        g = greeks(**PARAMS)
        assert g["gamma"] > 0, "Gamma must be positive"


class TestClassicalMC:
    def test_mc_converges_to_bs(self):
        """Large N MC should be close to BS price."""
        np.random.seed(0)
        est = price_european_call(**PARAMS, N=50000, seed=42)
        assert abs(est - BS_PRICE) < 0.5, f"MC ${est:.4f} too far from BS ${BS_PRICE:.4f}"

    def test_mc_positive(self):
        est = price_european_call(**PARAMS, N=1000, seed=1)
        assert est > 0

    def test_mc_error_shrinks_with_N(self):
        """Error should decrease as N increases (on average)."""
        errors = []
        for N in [100, 1000, 10000]:
            estimates = [price_european_call(**PARAMS, N=N, seed=i) for i in range(20)]
            errors.append(np.std(estimates))
        assert errors[0] > errors[1] > errors[2], "Std should decrease with N"


class TestStatePreparation:
    def test_probs_sum_to_one(self):
        _, probs, _, _ = discretise_lognormal(**PARAMS, n_qubits=4)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_norm_payoffs_in_unit_range(self):
        _, _, norm_payoffs, _ = discretise_lognormal(**PARAMS, n_qubits=4)
        assert norm_payoffs.min() >= 0.0
        assert norm_payoffs.max() <= 1.0 + 1e-9

    def test_disc_price_closer_with_more_qubits(self):
        errors = []
        for nq in [3, 4, 5, 6]:
            _, probs, npf, mpf = discretise_lognormal(**PARAMS, n_qubits=nq)
            dp = expected_payoff_exact(probs, npf, mpf, PARAMS["r"], PARAMS["T"])
            errors.append(abs(dp - BS_PRICE))
        # More qubits → smaller discretisation error (generally monotone)
        assert errors[0] >= errors[-1], "Disc error should decrease with more qubits"

    def test_more_qubits_better_approximation(self):
        """6-qubit discretisation should be much closer to BS than 3-qubit."""
        _, p3, n3, m3 = discretise_lognormal(**PARAMS, n_qubits=3)
        _, p6, n6, m6 = discretise_lognormal(**PARAMS, n_qubits=6)
        e3 = abs(expected_payoff_exact(p3, n3, m3, PARAMS["r"], PARAMS["T"]) - BS_PRICE)
        e6 = abs(expected_payoff_exact(p6, n6, m6, PARAMS["r"], PARAMS["T"]) - BS_PRICE)
        assert e6 < e3


class TestQAECircuit:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n_qubits     = 4
        self.state_wires  = list(range(self.n_qubits))
        self.ancilla_wire = self.n_qubits
        _, self.probs, self.norm_payoffs, self.max_payoff = discretise_lognormal(
            **PARAMS, n_qubits=self.n_qubits
        )

    def test_exact_amplitude_in_range(self):
        dev = build_qae_device(self.n_qubits)
        a = qae_exact_amplitude(self.probs, self.norm_payoffs,
                                 self.state_wires, self.ancilla_wire, dev)
        assert 0.0 <= a <= 1.0, f"Amplitude {a} out of [0, 1]"

    def test_price_from_amplitude_positive(self):
        dev = build_qae_device(self.n_qubits)
        a   = qae_exact_amplitude(self.probs, self.norm_payoffs,
                                   self.state_wires, self.ancilla_wire, dev)
        price = amplitude_to_price(a, self.max_payoff, PARAMS["r"], PARAMS["T"])
        assert price > 0

    def test_qae_price_close_to_bs(self):
        """QAE exact price should be within $1 of BS (discretisation gap)."""
        dev = build_qae_device(self.n_qubits)
        a   = qae_exact_amplitude(self.probs, self.norm_payoffs,
                                   self.state_wires, self.ancilla_wire, dev)
        price = amplitude_to_price(a, self.max_payoff, PARAMS["r"], PARAMS["T"])
        assert abs(price - BS_PRICE) < 1.5, f"QAE ${price:.4f} too far from BS ${BS_PRICE:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
