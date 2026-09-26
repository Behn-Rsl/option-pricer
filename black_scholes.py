"""Black–Scholes pricing and Greeks for European options on a non-dividend-paying stock.

Every function accepts plain floats or NumPy arrays (inputs broadcast against each
other), so the same code prices one option or a whole grid of scenarios.

Units used throughout
    time   years
    vol    decimal, so 0.20 means 20%
    rate   decimal, continuously compounded, so 0.05 means 5%
    vega   value change per 1 percentage point of volatility
    theta  value change per calendar day
    rho    value change per 1 percentage point of interest rate
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
from scipy.stats import norm

Kind = Literal["call", "put"]
Measure = Literal["price", "delta", "gamma", "vega", "theta", "rho"]

_EPS = 1e-10


@dataclass(frozen=True)
class Params:
    """One set of model inputs."""

    spot: float
    strike: float
    time: float
    vol: float
    rate: float

    def with_(self, **changes: float) -> "Params":
        return replace(self, **changes)


@dataclass(frozen=True)
class Result:
    """Prices, intermediate terms and Greeks for a call and a put."""

    d1: np.ndarray | float
    d2: np.ndarray | float
    discount: np.ndarray | float  # e^(-rT)
    call: np.ndarray | float
    put: np.ndarray | float
    delta_call: np.ndarray | float
    delta_put: np.ndarray | float
    gamma: np.ndarray | float
    vega: np.ndarray | float
    theta_call: np.ndarray | float
    theta_put: np.ndarray | float
    rho_call: np.ndarray | float
    rho_put: np.ndarray | float

    def get(self, measure: Measure, kind: Kind) -> np.ndarray | float:
        """Look up a value by name, e.g. result.get("delta", "put")."""
        if measure == "price":
            return self.call if kind == "call" else self.put
        if measure in ("gamma", "vega"):  # identical for calls and puts
            return getattr(self, measure)
        return getattr(self, f"{measure}_{kind}")


def black_scholes(spot, strike, time, vol, rate) -> Result:
    """Price a European call and put and compute their Greeks."""
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    T = np.maximum(np.asarray(time, dtype=float), _EPS)
    v = np.maximum(np.asarray(vol, dtype=float), _EPS)
    r = np.asarray(rate, dtype=float)

    sqrt_t = np.sqrt(T)
    vol_sqrt_t = v * sqrt_t
    d1 = (np.log(S / K) + (r + 0.5 * v**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    discount = np.exp(-r * T)

    n_d1 = norm.pdf(d1)
    N_d1, N_d2 = norm.cdf(d1), norm.cdf(d2)
    N_neg_d1, N_neg_d2 = norm.cdf(-d1), norm.cdf(-d2)

    call = S * N_d1 - K * discount * N_d2
    put = K * discount * N_neg_d2 - S * N_neg_d1
    time_decay = -S * n_d1 * v / (2 * sqrt_t)

    values = dict(
        d1=d1,
        d2=d2,
        discount=discount,
        call=call,
        put=put,
        delta_call=N_d1,
        delta_put=N_d1 - 1,
        gamma=n_d1 / (S * vol_sqrt_t),
        vega=S * n_d1 * sqrt_t / 100,
        theta_call=(time_decay - r * K * discount * N_d2) / 365,
        theta_put=(time_decay + r * K * discount * N_neg_d2) / 365,
        rho_call=K * T * discount * N_d2 / 100,
        rho_put=-K * T * discount * N_neg_d2 / 100,
    )
    # Hand back plain floats when every input was a scalar.
    return Result(**{k: float(x) if np.ndim(x) == 0 else x for k, x in values.items()})


def price(p: Params) -> Result:
    return black_scholes(p.spot, p.strike, p.time, p.vol, p.rate)


# ---------------------------------------------------------------------------
# Scenario shocks: which input moves the price most?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shock:
    name: str  # chart label, e.g. "Spot price"
    size_label: str  # e.g. "±10%"
    noun: str  # used in sentences, e.g. "spot price"
    phrase: str  # e.g. "a 10% move in spot"
    apply: Callable[[Params, int], Params]  # direction is -1 or +1


def _make_shocks(spot_pct, vol_pts, time_yrs, rate_pts, time_label, time_phrase):
    return (
        Shock("Spot price", f"±{spot_pct:g}%", "spot price", f"a {spot_pct:g}% move in spot",
              lambda p, d: p.with_(spot=p.spot * (1 + spot_pct / 100 * d))),
        Shock("Volatility", f"±{vol_pts:g} point{'s' if vol_pts != 1 else ''}", "volatility",
              f"a {vol_pts:g}-point move in volatility",
              lambda p, d: p.with_(vol=max(0.001, p.vol + vol_pts / 100 * d))),
        Shock("Time to expiry", time_label, "time to expiry", time_phrase,
              lambda p, d: p.with_(time=max(1 / 365, p.time + time_yrs * d))),
        Shock("Interest rate", f"±{rate_pts:g} point{'s' if rate_pts != 1 else ''}", "the interest rate",
              f"a {rate_pts:g}-point move in rates",
              lambda p, d: p.with_(rate=p.rate + rate_pts / 100 * d)),
    )


SHOCKS = {
    "small": _make_shocks(1, 1, 7 / 365, 0.25, "±1 week", "a week more or less to expiry"),
    "large": _make_shocks(10, 5, 0.25, 1, "±3 months", "three months more or less to expiry"),
}


@dataclass(frozen=True)
class Impact:
    shock: Shock
    lowered: float  # value change when the input goes down
    raised: float  # value change when the input goes up

    @property
    def biggest(self) -> float:
        return max(abs(self.lowered), abs(self.raised))


def shock_impacts(p: Params, kind: Kind, size: Literal["small", "large"] = "large") -> list[Impact]:
    """Bump each input down and up, and rank inputs by the largest change in value."""
    base = price(p).get("price", kind)
    impacts = [
        Impact(
            shock=s,
            lowered=price(s.apply(p, -1)).get("price", kind) - base,
            raised=price(s.apply(p, +1)).get("price", kind) - base,
        )
        for s in SHOCKS[size]
    ]
    return sorted(impacts, key=lambda i: i.biggest, reverse=True)


if __name__ == "__main__":
    r = black_scholes(spot=100, strike=100, time=1, vol=0.20, rate=0.05)
    print(f"Call {r.call:.4f}   Put {r.put:.4f}")
    for g in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"{g:>6}  call {r.get(g, 'call'):+.4f}   put {r.get(g, 'put'):+.4f}")
