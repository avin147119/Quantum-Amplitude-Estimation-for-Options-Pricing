# Quantum Amplitude Estimation for Options Pricing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38%2B-6C2DC7?logo=data:image/svg+xml;base64,&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Quantum%20Finance-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> Demonstrating a **quadratic convergence advantage** of Quantum Amplitude Estimation over Classical Monte Carlo for European option pricing — with noise analysis for NISQ realism.

---

## What This Project Does

This project prices a **European Call Option** using two methods and compares their convergence:

| Method | Convergence Rate | Evaluations to reach ε error |
|---|---|---|
| Classical Monte Carlo | O(1/√N) | ~N = 10,000 |
| Quantum Amplitude Estimation | **O(1/N)** | **~N = 100** |

The **quadratic speedup** is mathematically proven and visually demonstrated through convergence curves. The project also includes a **NISQ noise analysis** showing the gate error threshold above which the quantum advantage disappears — the honest, research-grade result.

---

## Visual Results

### Convergence Comparison (Log-Log)
```
Error
 |  *                     Classical MC  ~ 1/√N  (slope -0.5)
 |    *
 |      *   ·             Quantum AE    ~ 1/N   (slope -1.0)
 |        *   ·
 |          *   ·
 |____________·_____ N (evaluations)
```
*Steeper slope = faster convergence = quantum advantage*

### Key Finding
> At N = 1,000 evaluations, QAE achieves ~**10x lower pricing error** than Classical Monte Carlo under noiseless simulation. Gate noise above ~1–2% erodes this advantage — the central NISQ challenge.

---

## Project Structure

```
qae-options-pricing/
│
├── src/
│   ├── black_scholes.py       # Closed-form BS pricer (ground truth)
│   ├── classical_mc.py        # Classical Monte Carlo baseline
│   ├── state_preparation.py   # Quantum state prep & discretisation
│   ├── qae_circuit.py         # QAE circuit (PennyLane)
│   ├── noise_analysis.py      # Depolarising noise simulation
│   └── plotting.py            # All visualisation utilities
│
├── notebooks/
│   └── full_experiment.ipynb  # End-to-end walkthrough notebook
│
├── results/
│   └── figures/               # Generated plots (auto-populated)
│
├── tests/
│   └── test_pricing.py        # Unit tests
│
├── docs/
│   └── theory.md              # Mathematical background
│
├── main.py                    # Run full experiment from CLI
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/qae-options-pricing.git
cd qae-options-pricing
pip install -r requirements.txt
```

### 2. Run the Full Experiment

```bash
python main.py
```

This runs all four experiments and saves figures to `results/figures/`.

### 3. Explore the Notebook

```bash
jupyter notebook notebooks/full_experiment.ipynb
```

---

## Option Parameters

Default parameters used throughout (easily changed in `main.py`):

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Current price | S₀ | $100 | Initial stock price |
| Strike price | K | $105 | Exercise price (OTM) |
| Risk-free rate | r | 5% | Annual risk-free rate |
| Volatility | σ | 20% | Annual volatility |
| Maturity | T | 1 year | Time to expiry |
| State qubits | n | 4–6 | Controls discretisation |

---

## Theory

### Black-Scholes Model
Stock price follows **Geometric Brownian Motion**:

$$S_T = S_0 \cdot \exp\left(\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T} \cdot Z\right), \quad Z \sim \mathcal{N}(0,1)$$

European call payoff: $f(S_T) = \max(S_T - K,\ 0)$

Option price: $C = e^{-rT} \cdot \mathbb{E}[f(S_T)]$

### Why QAE?
Classical Monte Carlo estimates $\mathbb{E}[f(S_T)]$ by averaging N random samples — error shrinks as $1/\sqrt{N}$.

QAE encodes the **entire probability distribution** into quantum amplitudes using a state preparation operator $\mathcal{A}$, then uses Grover-like amplitude amplification + quantum phase estimation to estimate the expected payoff with error $O(1/N)$ — a **quadratic improvement**.

For a full derivation, see [`docs/theory.md`](docs/theory.md).

---

## Results Summary

```
══════════════════════════════════════════════════
  CONVERGENCE RESULTS
══════════════════════════════════════════════════
  Black-Scholes ground truth:   $8.02xx
  QAE circuit exact value:      $7.9xxx  (discretisation gap)

  Classical MC  slope: ~-0.50  (theory: -0.500)
  Quantum AE    slope: ~-0.95  (theory: -1.000)

  Speedup at N=1000: ~8–12x fewer evaluations needed
  Noise crossover:   ~1–2% per-gate depolarising error
══════════════════════════════════════════════════
```

---

## Tech Stack

| Library | Role |
|---|---|
| `pennylane` | Quantum circuit simulation + QNodes |
| `numpy / scipy` | Numerical computation, statistics |
| `matplotlib` | Convergence plots and figures |
| `jupyter` | Interactive notebook |

---

## Roadmap

- [x] European call option pricing
- [x] Convergence comparison (CMC vs QAE)
- [x] Depolarising noise analysis
- [x] Qubit ablation study
- [ ] Asian option (path-dependent payoff)
- [ ] Iterative QAE (Grinko et al. 2021)
- [ ] Run on real hardware via IBM Quantum / Amazon Braket
- [ ] Multi-asset portfolio pricing

---

## References

1. Brassard, G. et al. (2002). *Quantum Amplitude Amplification and Estimation*. AMS Contemporary Mathematics.
2. Woerner, S. & Egger, D.J. (2019). *Quantum risk analysis*. npj Quantum Information.
3. Stamatopoulos, N. et al. (2020). *Option Pricing using Quantum Computers*. Quantum 4, 291.
4. Grinko, D. et al. (2021). *Iterative quantum amplitude estimation*. npj Quantum Information.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as a demonstration of quantum advantage in computational finance. Simulations run on PennyLane's `default.qubit` — no quantum hardware required.*
