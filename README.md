# Black–Scholes option pricer

Prices European calls and puts on a non-dividend-paying stock and shows what the price is sensitive to:

- **Greeks** for both options, with a plain-English reading of each
- **What moves the price most**: bumps each input down and up and ranks them by impact
- **Spot × volatility heatmaps** of value, or profit and loss against the price you paid
- **One input at a time**: plot value or any Greek against spot, volatility, time or rates

## Files

| File | What it does |
| --- | --- |
| `black_scholes.py` | The model: prices, Greeks and shock scenarios. Works on floats or NumPy arrays. |
| `app.py` | The Streamlit interface. |
| `test_black_scholes.py` | Tests: textbook prices, put–call parity, Greeks against finite differences. |

## Run it in VS Code

1. Open this folder in VS Code (**File → Open Folder…**).
2. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   If VS Code asks whether to use the new environment for this workspace, choose **Yes**.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

   It opens in your browser at http://localhost:8501. Save a change to `app.py` and click **Rerun** in the
   browser (or press **R**) to see it.

You can also press **F5**: the included `.vscode/launch.json` runs the app under the debugger, so breakpoints
in `app.py` or `black_scholes.py` work.

## Run the tests

```bash
pytest
```

## Using the model on its own

```python
import black_scholes as bs

r = bs.black_scholes(spot=100, strike=100, time=1, vol=0.20, rate=0.05)
print(r.call, r.put, r.get("delta", "call"))
```

Running `python black_scholes.py` prints the default example.

## Assumptions

European exercise, no dividends, and constant volatility and interest rate over the option's life.
Vega and rho are per 1 percentage point; theta is per calendar day.
