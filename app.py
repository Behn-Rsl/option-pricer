"""Black–Scholes option pricer.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

import black_scholes as bs

st.set_page_config(page_title="Black–Scholes option pricer", page_icon="📈", layout="wide")

CALL, PUT = "#2459A6", "#8A3B7A"
GAIN, LOSS, NEUTRAL = "#1A7F64", "#C4452F", "#F7F9FA"
INK, MUTED = "#15233B", "#56667A"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
    h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif !important; font-weight: 600 !important; }
    .price { border-left: 3px solid var(--c); padding: 2px 0 6px 18px; margin-bottom: 8px; }
    .price-label { color: var(--c); font-weight: 600; font-size: .95rem; }
    .price-value { font-family: "Source Serif 4", Georgia, serif; font-size: 3.6rem; line-height: 1.05;
                   font-variant-numeric: lining-nums tabular-nums; }
    .price-meta { opacity: .72; font-size: .9rem; margin-top: 4px; }
    .summary { font-family: "Source Serif 4", Georgia, serif; font-size: 1.3rem; line-height: 1.45;
               max-width: 60ch; margin: 8px 0 4px; }
    .heat-title { font-family: "Source Serif 4", Georgia, serif; font-size: 1.2rem; font-weight: 600;
                  color: var(--c); margin: 12px 0 0; }
    [data-testid="stSliderThumbValue"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def minus(text: str) -> str:
    """Swap ASCII hyphens for proper minus signs in displayed numbers."""
    return text.replace("-", "\u2212")


def fmt(x: float, d: int = 2) -> str:
    return minus(f"{0.0 if abs(x) < 0.5 * 10**-d else x:.{d}f}")


def signed(x: float, d: int = 2) -> str:
    return minus(f"{0.0 if abs(x) < 0.5 * 10**-d else x:+.{d}f}")


def tint(hex_colour: str, amount: float) -> str:
    """Mix a colour with white. amount=1 keeps the colour, 0 gives white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * amount):02x}" for c in rgb)


# ---------------------------------------------------------------------------
# Inputs: each has a number box and a slider that stay in sync
# ---------------------------------------------------------------------------

FIELDS = {
    "spot": dict(label="Spot price  S", default=100.0, lo=1.0, hi=500.0, step=0.5, nmin=0.01, nmax=1e6, fmt="%.2f"),
    "strike": dict(label="Strike price  K", default=100.0, lo=1.0, hi=500.0, step=0.5, nmin=0.01, nmax=1e6, fmt="%.2f"),
    "time": dict(label="Time to expiry  T (years)", default=1.0, lo=0.01, hi=5.0, step=0.01, nmin=0.003, nmax=30.0, fmt="%.3f"),
    "vol": dict(label="Volatility  σ (%)", default=20.0, lo=1.0, hi=150.0, step=0.5, nmin=0.1, nmax=500.0, fmt="%.2f"),
    "rate": dict(label="Risk-free rate  r (%)", default=5.0, lo=-2.0, hi=20.0, step=0.1, nmin=-10.0, nmax=50.0, fmt="%.2f"),
}


def _from_number(name: str) -> None:
    f = FIELDS[name]
    st.session_state[f"{name}_slider"] = min(max(st.session_state[f"{name}_num"], f["lo"]), f["hi"])


def _from_slider(name: str) -> None:
    st.session_state[f"{name}_num"] = st.session_state[f"{name}_slider"]


def _reset_inputs() -> None:
    for name, f in FIELDS.items():
        st.session_state[f"{name}_num"] = f["default"]
        st.session_state[f"{name}_slider"] = f["default"]


with st.sidebar:
    st.header("Inputs")
    for name, f in FIELDS.items():
        st.session_state.setdefault(f"{name}_num", f["default"])
        st.session_state.setdefault(f"{name}_slider", f["default"])
        st.number_input(f["label"], min_value=f["nmin"], max_value=f["nmax"], step=f["step"], format=f["fmt"],
                        key=f"{name}_num", on_change=_from_number, args=(name,))
        st.slider(f["label"], min_value=f["lo"], max_value=f["hi"], step=f["step"], key=f"{name}_slider",
                  on_change=_from_slider, args=(name,), label_visibility="collapsed")
        if name == "time":
            days = round(st.session_state["time_num"] * 365)
            st.caption(f"About {days} day{'s' if days != 1 else ''}")
    st.button("Reset inputs", on_click=_reset_inputs)

params = bs.Params(
    spot=st.session_state["spot_num"],
    strike=st.session_state["strike_num"],
    time=st.session_state["time_num"],
    vol=st.session_state["vol_num"] / 100,
    rate=st.session_state["rate_num"] / 100,
)
res = bs.price(params)
S, K, T, VOL, R = params.spot, params.strike, params.time, params.vol, params.rate

# ---------------------------------------------------------------------------
# Header and prices
# ---------------------------------------------------------------------------

st.title("Black–Scholes option pricer")
st.caption("Prices European calls and puts on a non-dividend-paying stock, then shows which inputs "
           "that price depends on most. Change any input in the sidebar and everything updates.")


def moneyness(kind: str) -> str:
    if abs(S / K - 1) < 0.005:
        return "At the money"
    in_money = S > K if kind == "call" else S < K
    return "In the money" if in_money else "Out of the money"


def price_block(kind: str, value: float, colour: str) -> str:
    intrinsic = max(S - K, 0) if kind == "call" else max(K - S, 0)
    return (
        f'<div class="price" style="--c:{colour}">'
        f'<div class="price-label">{kind.capitalize()} value</div>'
        f'<div class="price-value">{fmt(value)}</div>'
        f'<div class="price-meta">{moneyness(kind)}. Intrinsic value {fmt(intrinsic)}, '
        f"time value {fmt(value - intrinsic)}.</div></div>"
    )


col_call, col_put = st.columns(2)
col_call.markdown(price_block("call", res.call, CALL), unsafe_allow_html=True)
col_put.markdown(price_block("put", res.put, PUT), unsafe_allow_html=True)

with st.expander("Show the worked calculation"):
    vol_sqrt_t = VOL * np.sqrt(T)
    k_disc = K * res.discount
    n_d1, n_d2 = norm.cdf(res.d1), norm.cdf(res.d2)
    st.caption("N(x) is the standard normal cumulative distribution: the probability that a standard "
               "normal variable comes out below x.")
    st.latex(
        r"\begin{aligned} d_1 &= \frac{\ln(S/K) + \left(r + \tfrac12\sigma^2\right)T}{\sigma\sqrt{T}} \\"
        rf"&= \frac{{\ln({S:.2f}/{K:.2f}) + ({R:.4f} + {0.5 * VOL**2:.4f}) \times {T:.3f}}}{{{vol_sqrt_t:.4f}}}"
        rf" = \mathbf{{{res.d1:.4f}}} \end{{aligned}}"
    )
    st.latex(rf"d_2 = d_1 - \sigma\sqrt{{T}} = {res.d1:.4f} - {vol_sqrt_t:.4f} = \mathbf{{{res.d2:.4f}}}")
    st.latex(
        r"\begin{aligned} C &= S\,N(d_1) - K e^{-rT} N(d_2) \\"
        rf"&= {S:.2f} \times {n_d1:.4f} - {k_disc:.4f} \times {n_d2:.4f}"
        rf" = \mathbf{{{res.call:.4f}}} \end{{aligned}}"
    )
    st.latex(
        r"\begin{aligned} P &= K e^{-rT} N(-d_2) - S\,N(-d_1) \\"
        rf"&= {k_disc:.4f} \times {1 - n_d2:.4f} - {S:.2f} \times {1 - n_d1:.4f}"
        rf" = \mathbf{{{res.put:.4f}}} \end{{aligned}}"
    )
    st.caption(f"Put–call parity check: C − P = {fmt(res.call - res.put, 4)}, "
               f"and S − Ke^(−rT) = {fmt(S - k_disc, 4)}.")

# ---------------------------------------------------------------------------
# Sensitivity views
# ---------------------------------------------------------------------------

tab_greeks, tab_drivers, tab_heat, tab_curve = st.tabs(
    ["Greeks", "What moves the price", "Spot × volatility heatmaps", "One input at a time"]
)

# --- Greeks table -----------------------------------------------------------
with tab_greeks:
    st.write("Each Greek measures how much the option's value moves when one input changes "
             "and the others stay fixed.")
    greeks = [
        ("Delta (Δ)", "delta", "Value change for a 1.00 rise in spot"),
        ("Gamma (Γ)", "gamma", "Change in delta for a 1.00 rise in spot"),
        ("Vega (ν)", "vega", "Value change for a 1-point rise in volatility"),
        ("Theta (Θ)", "theta", "Value change from one calendar day passing"),
        ("Rho (ρ)", "rho", "Value change for a 1-point rise in the interest rate"),
    ]
    table = pd.DataFrame(
        {
            "Greek": [g[0] for g in greeks],
            "Call": [res.get(g[1], "call") for g in greeks],
            "Put": [res.get(g[1], "put") for g in greeks],
            "Read it as": [g[2] for g in greeks],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Call": st.column_config.NumberColumn(format="%.4f"),
            "Put": st.column_config.NumberColumn(format="%.4f"),
        },
    )

# --- Tornado chart ----------------------------------------------------------
with tab_drivers:
    c1, c2 = st.columns(2)
    kind = c1.radio("Option", ["Call", "Put"], horizontal=True, key="drivers_kind").lower()
    size_label = c2.radio("Size of move", ["Small moves", "Large moves"], index=1, horizontal=True,
                          key="drivers_size")
    size = "small" if size_label.startswith("Small") else "large"

    impacts = bs.shock_impacts(params, kind, size)
    base_value = res.get("price", kind)
    top, runner_up = impacts[0], impacts[1]
    share = f", about {round(top.biggest / base_value * 100)}% of its current value" if base_value > 0.01 else ""
    st.markdown(
        f'<p class="summary">The {kind} is most sensitive to {top.shock.noun}: {top.shock.phrase} changes '
        f"its value by up to {fmt(top.biggest, 2 if top.biggest >= 0.01 else 4)}{share}. "
        f"{runner_up.shock.noun[0].upper() + runner_up.shock.noun[1:]} comes next.</p>",
        unsafe_allow_html=True,
    )

    colour = CALL if kind == "call" else PUT
    rows = impacts[::-1]  # Plotly draws the first bar at the bottom
    labels = [f"{i.shock.name}<br><span style='font-size:11px'>{i.shock.size_label}</span>" for i in rows]
    span = max(i.biggest for i in impacts) or 1
    fig = go.Figure()
    fig.add_bar(y=labels, x=[i.lowered for i in rows], orientation="h", name="Input lowered",
                marker_color=tint(colour, 0.38), text=[signed(i.lowered) for i in rows],
                textposition="outside", cliponaxis=False)
    fig.add_bar(y=labels, x=[i.raised for i in rows], orientation="h", name="Input raised",
                marker_color=colour, text=[signed(i.raised) for i in rows],
                textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(
        barmode="group", bargap=0.3, height=400, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="Change in option value", range=[-span * 1.3, span * 1.3], zeroline=False),
        hovermode=False,
    )
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("Each input is moved down and up on its own, with the others held fixed. "
               "Inputs are ranked by the biggest change in value.")

# --- Heatmaps ---------------------------------------------------------------
with tab_heat:
    st.write("Rows are volatility, columns are spot price, and the outlined cell is the one closest to your "
             "inputs. Switch to profit and loss to compare each scenario with what you paid.")

    def _clear_ranges() -> None:
        for k in ("heat_smin", "heat_smax", "heat_vmin", "heat_vmax"):
            st.session_state.pop(k, None)

    def _clear_paid() -> None:
        for k in ("paid_call", "paid_put"):
            st.session_state.pop(k, None)

    c_mode, c_range, c_paid = st.columns([1, 1.4, 1.2])
    mode = c_mode.radio("Show", ["Value", "Profit and loss"], horizontal=True, key="heat_mode")
    pnl = mode == "Profit and loss"

    with c_range:
        follow = st.toggle("Center ranges on current inputs", value=True, key="heat_follow", on_change=_clear_ranges)
        auto_smin, auto_smax = round(S * 0.8, 2), round(S * 1.2, 2)
        auto_vmin = round(max(1.0, VOL * 100 - 10), 1)
        auto_vmax = round(auto_vmin + 20, 1)
        if follow:
            smin, smax, vmin, vmax = auto_smin, auto_smax, auto_vmin, auto_vmax
            st.caption(f"Spot {fmt(smin)} to {fmt(smax)}, volatility {vmin:g}% to {vmax:g}%.")
        else:
            for k, v in (("heat_smin", auto_smin), ("heat_smax", auto_smax), ("heat_vmin", auto_vmin), ("heat_vmax", auto_vmax)):
                st.session_state.setdefault(k, v)
            a, b = st.columns(2)
            smin = a.number_input("Lowest spot", min_value=0.01, key="heat_smin")
            smax = b.number_input("Highest spot", min_value=0.01, key="heat_smax")
            vmin = a.number_input("Lowest volatility (%)", min_value=0.1, key="heat_vmin")
            vmax = b.number_input("Highest volatility (%)", min_value=0.1, key="heat_vmax")

    paid = {"call": res.call, "put": res.put}
    if pnl:
        with c_paid:
            use_fair = st.toggle("Use current fair values as price paid", value=True, key="paid_fair",
                                 on_change=_clear_paid)
            if use_fair:
                st.caption(f"Call paid {fmt(res.call)}, put paid {fmt(res.put)}.")
            else:
                st.session_state.setdefault("paid_call", round(res.call, 2))
                st.session_state.setdefault("paid_put", round(res.put, 2))
                a, b = st.columns(2)
                paid["call"] = a.number_input("Call paid", min_value=0.0, key="paid_call")
                paid["put"] = b.number_input("Put paid", min_value=0.0, key="paid_put")

    if not (smax > smin > 0 and vmax > vmin > 0):
        st.warning("Keep each lower bound below its upper bound, and keep spot and volatility above zero.")
    else:
        n = 11  # odd, so the current inputs land in the middle cell when ranges are centred
        spots = np.linspace(smin, smax, n)
        vols_pct = np.linspace(vmin, vmax, n)
        grid = bs.black_scholes(spots[None, :], K, T, vols_pct[:, None] / 100, R)
        ds, dv = spots[1] - spots[0], vols_pct[1] - vols_pct[0]
        col_j = int(round((S - smin) / ds)) if smin - ds / 2 <= S <= smax + ds / 2 else None
        row_i = int(round((VOL * 100 - vmin) / dv)) if vmin - dv / 2 <= VOL * 100 <= vmax + dv / 2 else None
        spot_dec = 0 if smax >= 1000 else 1 if smax >= 100 else 2

        def cell_text(x: float) -> str:
            return fmt(x, 0 if abs(x) >= 100 else 1 if abs(x) >= 10 else 2)

        for column, kind, colour in zip(st.columns(2), ("call", "put"), (CALL, PUT)):
            z = grid.get("price", kind) - (paid[kind] if pnl else 0)
            if pnl:
                scale, zmid = [[0, LOSS], [0.5, NEUTRAL], [1, GAIN]], 0
            else:
                scale, zmid = [[0, NEUTRAL], [1, colour]], None
            fig = go.Figure(
                go.Heatmap(
                    z=z, x=spots, y=vols_pct, colorscale=scale, zmid=zmid,
                    text=np.vectorize(cell_text)(z), texttemplate="%{text}", textfont=dict(size=11),
                    xgap=2, ygap=2,
                    hovertemplate="Spot %{x:.2f}<br>Volatility %{y:.2f}%<br>"
                    + ("Profit and loss" if pnl else "Value") + " %{z:.4f}<extra></extra>",
                    colorbar=dict(thickness=10, len=0.9, outlinewidth=0),
                )
            )
            if row_i is not None and col_j is not None:
                fig.add_shape(type="rect", line=dict(color=INK, width=2.5),
                              x0=spots[col_j] - ds / 2, x1=spots[col_j] + ds / 2,
                              y0=vols_pct[row_i] - dv / 2, y1=vols_pct[row_i] + dv / 2)
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_xaxes(title="Spot price", tickvals=spots, ticktext=[fmt(s, spot_dec) for s in spots],
                             showgrid=False, tickangle=0)
            fig.update_yaxes(title="Volatility (%)", tickvals=vols_pct, ticktext=[f"{v:g}" for v in np.round(vols_pct, 1)],
                             showgrid=False)
            with column:
                title = "profit and loss" if pnl else "value"
                st.markdown(f'<p class="heat-title" style="--c:{colour}">{kind.capitalize()} {title}</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig, config={"displayModeBar": False}, key=f"heat_{kind}")

# --- Curve explorer ---------------------------------------------------------
with tab_curve:
    X_OPTIONS = {
        "Spot price": dict(field="spot", scale=1, unit="", axis="Spot price",
                           span=lambda p: (max(0.01, p.spot * 0.5), p.spot * 1.5)),
        "Volatility": dict(field="vol", scale=100, unit="%", axis="Volatility (%)",
                           span=lambda p: (0.01, max(1.0, p.vol * 2))),
        "Time to expiry": dict(field="time", scale=1, unit=" yrs", axis="Time to expiry (years)",
                               span=lambda p: (1 / 365, max(2.0, p.time * 2))),
        "Interest rate": dict(field="rate", scale=100, unit="%", axis="Interest rate (%)",
                              span=lambda p: (min(0.0, p.rate - 0.02), max(0.15, p.rate * 2 + 0.02))),
    }
    Y_OPTIONS = {
        "Value": ("price", "Option value in the same currency as the spot price."),
        "Delta": ("delta", "How much the value moves for a 1.00 rise in spot. Calls run from 0 to 1, puts from −1 to 0."),
        "Gamma": ("gamma", "How fast delta changes as spot moves. It is the same for calls and puts, so the lines overlap."),
        "Vega": ("vega", "Value change for a 1-point rise in volatility. It is the same for calls and puts, so the lines overlap."),
        "Theta": ("theta", "Value lost or gained as one calendar day passes, with everything else fixed."),
        "Rho": ("rho", "Value change for a 1-point rise in the interest rate."),
    }
    c1, c2 = st.columns(2)
    y_name = c1.selectbox("Plot", list(Y_OPTIONS), key="curve_y")
    x_name = c2.selectbox("Against", list(X_OPTIONS), key="curve_x")
    measure, note = Y_OPTIONS[y_name]
    xo = X_OPTIONS[x_name]
    st.caption(note)

    lo, hi = xo["span"](params)
    xs = np.linspace(lo, hi, 240)
    args = dict(spot=S, strike=K, time=T, vol=VOL, rate=R)
    args[xo["field"]] = xs
    curve = bs.black_scholes(**args)
    x_show = xs * xo["scale"]
    decimals = 2 if measure == "price" else 4
    overlap = measure in ("gamma", "vega")

    fig = go.Figure()
    if measure == "price" and xo["field"] == "spot":
        fig.add_scatter(x=x_show, y=np.maximum(xs - K, 0), name="Call value at expiry", hoverinfo="skip", legendrank=3,
                        line=dict(color=CALL, width=1.5, dash="dot"), opacity=0.5)
        fig.add_scatter(x=x_show, y=np.maximum(K - xs, 0), name="Put value at expiry", hoverinfo="skip", legendrank=4,
                        line=dict(color=PUT, width=1.5, dash="dot"), opacity=0.5)
    fig.add_scatter(x=x_show, y=curve.get(measure, "call"), name="Call", legendrank=1, line=dict(color=CALL, width=3),
                    hovertemplate=f"%{{y:.{decimals}f}}")
    fig.add_scatter(x=x_show, y=curve.get(measure, "put"), name="Put", legendrank=2,
                    line=dict(color=PUT, width=3, dash="dash" if overlap else "solid"),
                    hovertemplate=f"%{{y:.{decimals}f}}")
    x_now = getattr(params, xo["field"]) * xo["scale"]
    fig.add_vline(x=x_now, line_color=MUTED, line_width=1, line_dash="dot")
    fig.add_scatter(x=[x_now, x_now], y=[res.get(measure, "call"), res.get(measure, "put")], mode="markers",
                    marker=dict(color=[CALL, PUT], size=10, line=dict(color="white", width=2)),
                    name="Your inputs", hoverinfo="skip", showlegend=False)
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title=xo["axis"], hoverformat=".2f"), yaxis=dict(title=y_name),
    )
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.markdown(
        f"At your inputs ({x_name.lower()} {x_now:.2f}{xo['unit']}): call **{fmt(res.get(measure, 'call'), decimals)}**, "
        f"put **{fmt(res.get(measure, 'put'), decimals)}**."
    )

st.divider()
st.caption("Assumes European exercise, no dividends, and constant volatility and interest rate over the "
           "option's life. Vega and rho are per 1 percentage point; theta is per calendar day.")
