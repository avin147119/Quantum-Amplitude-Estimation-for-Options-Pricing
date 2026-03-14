# Mathematical Background

## 1. European Call Option

A **European call option** gives the holder the right (not obligation) to buy an asset at price $K$ (strike) at time $T$ (maturity).

**Payoff at maturity:**
$$f(S_T) = \max(S_T - K,\ 0)$$

**Option price** (risk-neutral pricing):
$$C = e^{-rT} \cdot \mathbb{E}^\mathbb{Q}[f(S_T)]$$

where $r$ is the risk-free rate and $\mathbb{E}^\mathbb{Q}$ is the expectation under the risk-neutral measure.

---

## 2. Geometric Brownian Motion

Under the Black-Scholes model, the stock price follows:
$$dS_t = r S_t\, dt + \sigma S_t\, dW_t$$

with solution:
$$S_T = S_0 \cdot \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\cdot Z\right], \quad Z \sim \mathcal{N}(0,1)$$

So $\ln S_T \sim \mathcal{N}\!\left(\mu_\ell,\, \sigma_\ell^2\right)$ with:
$$\mu_\ell = \ln S_0 + \left(r - \frac{\sigma^2}{2}\right)T, \qquad \sigma_\ell = \sigma\sqrt{T}$$

---

## 3. Classical Monte Carlo

**Algorithm:**
1. Draw $N$ i.i.d. samples $Z_1, \ldots, Z_N \sim \mathcal{N}(0,1)$
2. Compute $S_T^{(i)} = S_0 \exp\!\left[(r - \sigma^2/2)T + \sigma\sqrt{T}Z_i\right]$
3. Compute $f_i = \max(S_T^{(i)} - K,\ 0)$
4. Estimate: $\hat{C} = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^N f_i$

**Error:** By the Central Limit Theorem:
$$\text{RMSE}(\hat{C}) = \frac{\text{Std}[f(S_T)]}{\sqrt{N}} \propto \frac{1}{\sqrt{N}}$$

This $O(1/\sqrt{N})$ convergence is the fundamental limitation of classical Monte Carlo.

---

## 4. Quantum Amplitude Estimation

### 4.1 Setup

Discretise the log-normal distribution into $M = 2^n$ bins. Let:
- $p_i$ = probability of bin $i$
- $f_i \in [0, 1]$ = normalised payoff of bin $i$ (scaled by $f_{\max}$)

### 4.2 State preparation operator $\mathcal{A}$

$$\mathcal{A}|0\rangle^{\otimes n}|0\rangle = \sum_{i=0}^{M-1} \sqrt{p_i}\,|i\rangle \otimes \left(\sqrt{f_i}\,|1\rangle + \sqrt{1-f_i}\,|0\rangle\right)$$

The probability of measuring $|1\rangle$ on the ancilla qubit is:
$$a = \sum_{i=0}^{M-1} p_i f_i = \mathbb{E}[\text{norm. payoff}]$$

The true option price is recovered as:
$$C = e^{-rT} \cdot a \cdot f_{\max}$$

### 4.3 Grover operator $\mathcal{Q}$

$$\mathcal{Q} = \mathcal{A}\, S_0\, \mathcal{A}^\dagger\, S_\chi$$

where:
- $S_\chi$ = reflection about the "good" subspace (ancilla = $|1\rangle$): flips phase of good states
- $S_0$ = reflection about $|0\rangle^{\otimes(n+1)}$: flips phase of the all-zeros state

### 4.4 Quantum Phase Estimation

After $m$ Grover iterations:
$$\mathcal{Q}^m \mathcal{A}|0\rangle = \sin\!\left[(2m+1)\theta\right]|\text{good}\rangle + \cos\!\left[(2m+1)\theta\right]|\text{bad}\rangle$$

where $\sin^2\!\theta = a$ (the amplitude we want to estimate).

QPE extracts $\theta$, from which $a = \sin^2\!\theta$ is recovered.

### 4.5 Convergence rate

With $N$ total oracle evaluations:

| Method | Error | Evaluations for error $\varepsilon$ |
|---|---|---|
| Classical MC | $O(1/\sqrt{N})$ | $O(1/\varepsilon^2)$ |
| Quantum AE | $O(1/N)$ | $O(1/\varepsilon)$ |

This is a **quadratic speedup** — for the same number of evaluations, QAE achieves error $\varepsilon$ instead of $\varepsilon^{1/2}$.

---

## 5. NISQ Noise

On current quantum hardware, every gate introduces errors. Depolarising noise with probability $p$ per gate applies a random Pauli operator, gradually destroying the coherence that QAE relies on.

The crossover condition — where noise fully erodes the quantum advantage — depends on:
- Circuit depth (number of Grover iterations $m$)
- Per-gate error probability $p$
- Number of qubits $n$

Empirically, the advantage persists for $p \lesssim 1\%$ in small circuits, but degrades rapidly beyond that threshold.

---

## References

1. Brassard, G., Høyer, P., Mosca, M., & Tapp, A. (2002). Quantum amplitude amplification and estimation. *AMS Contemporary Mathematics*, 305, 53–74.

2. Woerner, S., & Egger, D. J. (2019). Quantum risk analysis. *npj Quantum Information*, 5(1), 15.

3. Stamatopoulos, N., Egger, D. J., Sun, Y., Zoufal, C., Iten, R., Shen, N., & Woerner, S. (2020). Option pricing using quantum computers. *Quantum*, 4, 291.

4. Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2021). Iterative quantum amplitude estimation. *npj Quantum Information*, 7(1), 52.
