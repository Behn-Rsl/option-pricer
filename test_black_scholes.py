"""Run with:  pytest"""

import numpy as np
import pytest

import black_scholes as bs

CASES = [
    (100, 100, 1.0, 0.20, 0.05),
    (80, 100, 0.5, 0.35, 0.02),
    (120, 90, 2.0, 0.15, 0.00),
    (50, 60, 0.1, 0.60, 0.08),
]


def test_textbook_values():
    r = bs.black_scholes(100, 100, 1, 0.20, 0.05)
    assert r.call == pytest.approx(10.4506, abs=1e-4)
    assert r.put == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,T,v,r", CASES)
def test_put_call_parity(S, K, T, v, r):
    res = bs.black_scholes(S, K, T, v, r)
    assert res.call - res.put == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,v,r", CASES)
def test_greeks_match_finite_differences(kind, S, K, T, v, r):
    def value(**bump):
        args = dict(spot=S, strike=K, time=T, vol=v, rate=r)
        args.update({k: args[k] + dv for k, dv in bump.items()})
        return bs.black_scholes(**args).get("price", kind)

    res = bs.black_scholes(S, K, T, v, r)
    h = 1e-3 * S
    approx = pytest.approx

    assert res.get("delta", kind) == approx((value(spot=h) - value(spot=-h)) / (2 * h), rel=1e-4, abs=1e-8)
    assert res.get("gamma", kind) == approx((value(spot=h) - 2 * value() + value(spot=-h)) / h**2, rel=1e-3, abs=1e-6)
    assert res.get("vega", kind) == approx((value(vol=1e-5) - value(vol=-1e-5)) / 2e-5 / 100, rel=1e-5, abs=1e-8)
    assert res.get("theta", kind) == approx(-(value(time=1e-5) - value(time=-1e-5)) / 2e-5 / 365, rel=1e-4, abs=1e-8)
    assert res.get("rho", kind) == approx((value(rate=1e-5) - value(rate=-1e-5)) / 2e-5 / 100, rel=1e-5, abs=1e-8)


def test_grid_matches_scalar_pricing():
    spots = np.linspace(80, 120, 11)
    vols = np.linspace(0.10, 0.30, 11)
    grid = bs.black_scholes(spots[None, :], 100, 1, vols[:, None], 0.05)
    assert grid.call.shape == (11, 11)
    assert grid.call[3, 7] == pytest.approx(bs.black_scholes(spots[7], 100, 1, vols[3], 0.05).call)


def test_spot_is_the_biggest_driver_for_an_at_the_money_call():
    p = bs.Params(spot=100, strike=100, time=1, vol=0.20, rate=0.05)
    impacts = bs.shock_impacts(p, "call", "large")
    assert impacts[0].shock.name == "Spot price"
    assert impacts[0].raised > 0 > impacts[0].lowered
