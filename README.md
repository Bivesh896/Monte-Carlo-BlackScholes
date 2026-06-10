# Monte Carlo + Black-Scholes Option Pricing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A complete quantitative finance toolkit for pricing European and American options using two methods: the closed-form **Black-Scholes** formula and **Monte Carlo simulation** with variance reduction. Also includes Greeks computation (Delta, Gamma, Theta, Vega, Rho) and implied volatility solver via Newton-Raphson.

---

## Theory

### Black-Scholes (Closed Form)

Assumes log-normally distributed asset prices under risk-neutral measure:

```
S_T = S_0 · exp((r - σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)

Call: C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
Put:  P = K·e^(-rT)·N(-d₂) - S₀·N(-d₁)

d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

### Monte Carlo Simulation

Simulates `N` price paths using geometric Brownian motion, then discounts expected payoff:

```
C ≈ e^(-rT) · E[max(S_T - K, 0)]
```

Variance reduction via **antithetic variates**: for each random path `Z`, also simulate `–Z` and average — halves variance without doubling paths.

---

## Features

- European Call & Put: Black-Scholes + Monte Carlo
- American Put: Least-Squares Monte Carlo (Longstaff-Schwartz)
- All five Greeks: Δ, Γ, Θ, V (vega), ρ
- Implied volatility solver (Newton-Raphson, converges in ~10 iters)
- Convergence plot: MC price vs number of paths
- Payoff diagram generator

---

## Quickstart

```bash
pip install -r requirements.txt

# Price a European call
python options_pricing.py --mode european --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --option call

# Compare BS vs MC across strikes
python options_pricing.py --mode compare

# Compute Greeks
python options_pricing.py --mode greeks --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0

# Solve implied volatility
python options_pricing.py --mode iv --market_price 10.45 --S 100 --K 100 --r 0.05 --T 1.0

# Longstaff-Schwartz American Put
python options_pricing.py --mode american --S 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0
```

---

## Sample Output

```
European Call  (S=100, K=100, r=5%, σ=20%, T=1Y)
─────────────────────────────────────────────────
Black-Scholes:  10.4506
Monte Carlo:    10.4318  (95% CI: [10.37, 10.49])
MC Std Error:   0.0309
Antithetic:     10.4491  (std error: 0.0212, 31% var reduction)

Greeks (Black-Scholes)
Delta:  0.6368
Gamma:  0.0188
Theta: -6.4140 (per year)
Vega:   37.524
Rho:    53.232
```

---

## Project Structure

```
options-pricing/
├── options_pricing.py    # Full implementation
├── requirements.txt
└── README.md
```

---

## References

- Black & Scholes (1973) — *The Pricing of Options and Corporate Liabilities*
- Longstaff & Schwartz (2001) — *Valuing American Options by Simulation*
- Hull (2018) — *Options, Futures, and Other Derivatives* (10th ed.)
