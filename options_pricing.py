"""
Monte Carlo + Black-Scholes Option Pricing
Covers: European calls/puts, American put (Longstaff-Schwartz),
        Greeks, implied volatility, antithetic variance reduction.

Author: Bivesh Kumar Dalai
"""

import argparse
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ─────────────────────────────────────────────────────────────
# Black-Scholes closed form
# ─────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    """Compute d1 and d2 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, r: float, sigma: float, T: float,
             option: str = "call") -> float:
    """Black-Scholes price for European option."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    disc = np.exp(-r * T)
    if option == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, r: float, sigma: float, T: float,
              option: str = "call") -> dict:
    """Compute all five Greeks analytically."""
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    disc = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T  # per 1 unit change in sigma

    if option == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        )
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        )
        rho = -K * T * disc * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,  # per year; divide by 365 for daily
        "vega": vega,
        "rho": rho,
    }


# ─────────────────────────────────────────────────────────────
# Monte Carlo (European, with antithetic variates)
# ─────────────────────────────────────────────────────────────

def mc_european(
    S: float, K: float, r: float, sigma: float, T: float,
    option: str = "call",
    n_paths: int = 200_000,
    seed: int = 42,
    antithetic: bool = True,
) -> dict:
    """
    Monte Carlo price for European option.
    Uses geometric Brownian motion terminal value directly.
    Antithetic variates: pairs (Z, -Z) halve variance.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2

    Z = rng.standard_normal(half)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T)

    ST_pos = S * np.exp(drift + diffusion * Z)
    ST_neg = S * np.exp(drift - diffusion * Z)  # antithetic

    if option == "call":
        payoffs_pos = np.maximum(ST_pos - K, 0)
        payoffs_neg = np.maximum(ST_neg - K, 0)
    else:
        payoffs_pos = np.maximum(K - ST_pos, 0)
        payoffs_neg = np.maximum(K - ST_neg, 0)

    if antithetic:
        payoffs = 0.5 * (payoffs_pos + payoffs_neg)
    else:
        payoffs = np.concatenate([payoffs_pos, payoffs_neg])

    disc = np.exp(-r * T)
    discounted = disc * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(len(discounted))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr

    return {
        "price": price,
        "stderr": stderr,
        "ci": (ci_low, ci_high),
        "n_paths": n_paths,
    }


# ─────────────────────────────────────────────────────────────
# Longstaff-Schwartz American Put
# ─────────────────────────────────────────────────────────────

def lsmc_american_put(
    S: float, K: float, r: float, sigma: float, T: float,
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: int = 42,
) -> float:
    """
    Longstaff-Schwartz Monte Carlo for American put.
    Regresses continuation value on basis functions of S.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Simulate price paths: shape (n_steps+1, n_paths)
    Z = rng.standard_normal((n_steps, n_paths))
    log_S = np.zeros((n_steps + 1, n_paths))
    log_S[0] = np.log(S)
    for t in range(1, n_steps + 1):
        log_S[t] = log_S[t - 1] + (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
    paths = np.exp(log_S)

    # Cash flows at maturity
    cash_flow = np.maximum(K - paths[-1], 0)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        St = paths[t]
        intrinsic = np.maximum(K - St, 0)
        in_money = intrinsic > 0

        if in_money.sum() == 0:
            cash_flow *= disc
            continue

        # Regression: continuation value ~ a + b*S + c*S²
        X = St[in_money]
        basis = np.column_stack([np.ones_like(X), X, X ** 2])
        Y = cash_flow[in_money] * disc

        coeffs, *_ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation = basis @ coeffs

        # Exercise if intrinsic > continuation
        exercise = intrinsic[in_money] > continuation
        cash_flow[in_money] = np.where(exercise, intrinsic[in_money], cash_flow[in_money] * disc)
        cash_flow[~in_money] *= disc

    return (cash_flow * disc).mean()


# ─────────────────────────────────────────────────────────────
# Implied Volatility (Brent's method wrapper)
# ─────────────────────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    S: float, K: float, r: float, T: float,
    option: str = "call",
) -> float:
    """Solve for σ such that BS_price(σ) = market_price using Brent's method."""
    def objective(sigma):
        return bs_price(S, K, r, sigma, T, option) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-8)
    except ValueError:
        iv = float("nan")
    return iv


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def mode_european(args):
    bs = bs_price(args.S, args.K, args.r, args.sigma, args.T, args.option)
    mc = mc_european(args.S, args.K, args.r, args.sigma, args.T, args.option)
    print(f"\nEuropean {args.option.upper()}  (S={args.S}, K={args.K}, r={args.r}, σ={args.sigma}, T={args.T}Y)")
    print("─" * 55)
    print(f"Black-Scholes : {bs:.4f}")
    print(f"Monte Carlo   : {mc['price']:.4f}  (95% CI: [{mc['ci'][0]:.2f}, {mc['ci'][1]:.2f}])")
    print(f"MC Std Error  : {mc['stderr']:.4f}")


def mode_greeks(args):
    g = bs_greeks(args.S, args.K, args.r, args.sigma, args.T, args.option)
    print(f"\nGreeks (Black-Scholes) — {args.option.upper()}")
    print("─" * 35)
    for name, val in g.items():
        print(f"{name.capitalize():8s}: {val:.4f}")


def mode_iv(args):
    iv = implied_volatility(args.market_price, args.S, args.K, args.r, args.T, args.option)
    print(f"\nImplied Volatility: {iv:.4%}")
    check = bs_price(args.S, args.K, args.r, iv, args.T, args.option)
    print(f"BS price at IV    : {check:.4f}  (market: {args.market_price:.4f})")


def mode_american(args):
    price = lsmc_american_put(args.S, args.K, args.r, args.sigma, args.T)
    euro_put = bs_price(args.S, args.K, args.r, args.sigma, args.T, "put")
    print(f"\nAmerican Put (LSMC)  : {price:.4f}")
    print(f"European Put (BS)    : {euro_put:.4f}")
    print(f"Early Exercise Premium: {price - euro_put:.4f}")


def mode_compare(args):
    strikes = np.arange(80, 121, 5)
    print(f"\n{'K':>6} {'BS Call':>10} {'MC Call':>10} {'Diff':>8}")
    print("─" * 40)
    for K in strikes:
        bs = bs_price(args.S, K, args.r, args.sigma, args.T, "call")
        mc = mc_european(args.S, K, args.r, args.sigma, args.T, "call", n_paths=100_000)["price"]
        print(f"{K:>6} {bs:>10.4f} {mc:>10.4f} {abs(bs - mc):>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option Pricing: Black-Scholes + Monte Carlo")
    parser.add_argument("--mode", choices=["european", "greeks", "iv", "american", "compare"], default="european")
    parser.add_argument("--S", type=float, default=100.0, help="Spot price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--T", type=float, default=1.0, help="Time to expiry (years)")
    parser.add_argument("--option", choices=["call", "put"], default="call")
    parser.add_argument("--market_price", type=float, default=10.45, help="Market price (for IV mode)")
    args = parser.parse_args()

    dispatch = {
        "european": mode_european,
        "greeks": mode_greeks,
        "iv": mode_iv,
        "american": mode_american,
        "compare": mode_compare,
    }
    dispatch[args.mode](args)
