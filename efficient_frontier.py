"""
Efficient Frontier Calculator with IBKR Portfolio Import
=========================================================
Requirements:
    pip install yfinance scipy numpy pandas matplotlib seaborn

Usage:
    # 1. With your IBKR CSV export:
    python efficient_frontier.py --ibkr portfolio.csv

    # 2. With manual tickers:
    python efficient_frontier.py --tickers AAPL MSFT GOOGL AMZN TSLA

    # 3. IBKR CSV + extra tickers to explore:
    python efficient_frontier.py --ibkr portfolio.csv --tickers NVDA BRK-B

IBKR CSV Export Instructions:
    - Log in to IBKR Client Portal or TWS
    - Go to Reports → Flex Queries (or Activity → Statements)
    - Export a "Portfolio" or "Open Positions" report as CSV
    - The script auto-detects the format and extracts tickers

Author: Generated with Claude
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("❌  yfinance not found. Run: pip install yfinance scipy numpy pandas matplotlib seaborn")
    sys.exit(1)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOOKBACK_YEARS = 12          # Historical data window
RISK_FREE_RATE = 0.045      # Annual risk-free rate (update to current T-bill rate)
N_PORTFOLIOS = 8000         # Monte Carlo simulations for frontier
N_FRONTIER_POINTS = 200     # Points on the efficient frontier curve
TRADING_DAYS = 252


# ─────────────────────────────────────────────
# IBKR CSV PARSER
# ─────────────────────────────────────────────
def parse_ibkr_csv(filepath: str) -> pd.DataFrame:
    """
    Parse IBKR portfolio CSV exports.
    Handles both Flex Query format and Activity Statement format.
    Returns a DataFrame with columns: ticker, quantity, market_value, currency
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"\n📂  Parsing IBKR file: {filepath}")

    raw = pd.read_csv(filepath, header=None, dtype=str)

    # ── Try Flex Query format (section-based, comma-delimited) ──
    # Flex exports have rows like: "OpenPositions,Header,..." and "OpenPositions,Data,..."
    if raw.iloc[:, 0].str.contains("OpenPositions|Position|Positions", case=False, na=False).any():
        return _parse_flex_query(raw)

    # ── Try Activity Statement format ──
    # These have a "Positions" section with a header row
    if raw.iloc[:, 0].str.contains("Positions|Open Positions", case=False, na=False).any():
        return _parse_activity_statement(raw)

    # ── Fallback: treat as simple CSV with standard columns ──
    return _parse_simple_csv(filepath)


def _parse_flex_query(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse IBKR Flex Query CSV format."""
    # Find the OpenPositions section
    mask = raw.iloc[:, 0].str.contains("OpenPositions", case=False, na=False)
    section = raw[mask].copy()

    header_row = section[section.iloc[:, 1].str.lower() == "header"]
    data_rows = section[section.iloc[:, 1].str.lower() == "data"]

    if header_row.empty:
        raise ValueError("Could not find OpenPositions header in Flex Query CSV")

    headers = header_row.iloc[0].tolist()
    df = data_rows.copy()
    df.columns = range(len(df.columns))

    result = pd.DataFrame()
    col_map = {h.lower(): i for i, h in enumerate(headers)}

    ticker_col = next((col_map[k] for k in ["symbol", "ticker"] if k in col_map), None)
    qty_col = next((col_map[k] for k in ["position", "quantity", "qty"] if k in col_map), None)
    mv_col = next((col_map[k] for k in ["positionvalue", "marketvalue", "market value"] if k in col_map), None)
    curr_col = next((col_map[k] for k in ["currency", "curr"] if k in col_map), None)
    asset_col = next((col_map[k] for k in ["assetclass", "asset class", "sectype"] if k in col_map), None)

    if ticker_col is None:
        raise ValueError("Could not find ticker/symbol column in Flex Query CSV")

    result["ticker"] = df.iloc[:, ticker_col].str.strip().values
    result["quantity"] = pd.to_numeric(df.iloc[:, qty_col].str.replace(",", ""), errors="coerce") if qty_col else np.nan
    result["market_value"] = pd.to_numeric(df.iloc[:, mv_col].str.replace(",", ""), errors="coerce") if mv_col else np.nan
    result["currency"] = df.iloc[:, curr_col].str.strip().values if curr_col else "USD"

    # Filter to equity only
    if asset_col:
        result["asset_class"] = df.iloc[:, asset_col].str.strip().values
        result = result[result["asset_class"].str.upper().isin(["STK", "STOCK", "EQUITY", "EQY"])]

    result = result[result["ticker"].notna() & (result["ticker"] != "")]
    print(f"   ✓ Flex Query format — found {len(result)} positions")
    return result.reset_index(drop=True)


def _parse_activity_statement(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse IBKR Activity Statement CSV format."""
    positions_start = None
    header_idx = None

    for i, row in raw.iterrows():
        cell = str(row.iloc[0]).strip()
        if "Positions" in cell or "Open Positions" in cell:
            positions_start = i
        if positions_start and i > positions_start:
            if "Symbol" in row.values or "Ticker" in row.values:
                header_idx = i
                break
            if i > positions_start + 5:
                break

    if header_idx is None:
        raise ValueError("Could not find Positions section in Activity Statement CSV")

    headers = raw.iloc[header_idx].tolist()
    col_map = {str(h).strip().lower(): i for i, h in enumerate(headers)}

    data = raw.iloc[header_idx + 1:].copy()
    # Stop at next section
    stop = data[data.iloc[:, 0].str.contains("^[A-Z]", na=False) & 
                ~data.iloc[:, 0].str.match(r"^[A-Z]{1,5}$")].index
    if len(stop):
        data = data.loc[:stop[0] - 1]

    result = pd.DataFrame()
    ticker_col = next((col_map[k] for k in ["symbol", "ticker"] if k in col_map), None)
    qty_col = next((col_map[k] for k in ["quantity", "position", "qty"] if k in col_map), None)
    mv_col = next((col_map[k] for k in ["value in usd", "market value", "value", "market_value"] if k in col_map), None)

    if ticker_col is None:
        raise ValueError("Could not locate ticker column in Activity Statement")

    result["ticker"] = data.iloc[:, ticker_col].str.strip().values
    result["quantity"] = pd.to_numeric(data.iloc[:, qty_col].str.replace(",", ""), errors="coerce") if qty_col else np.nan
    result["market_value"] = pd.to_numeric(data.iloc[:, mv_col].str.replace(",", ""), errors="coerce") if mv_col else np.nan
    result["currency"] = "USD"

    result = result[result["ticker"].notna() & result["ticker"].str.match(r"^[A-Z]{1,5}$")]
    print(f"   ✓ Activity Statement format — found {len(result)} positions")
    return result.reset_index(drop=True)


def _parse_simple_csv(filepath: str) -> pd.DataFrame:
    """Parse a simple CSV: expects columns Symbol/Ticker, Quantity, MarketValue."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    ticker_col = next((c for c in df.columns if c in ["symbol", "ticker", "stock"]), None)
    qty_col = next((c for c in df.columns if "qty" in c or "quantity" in c or "position" in c), None)
    mv_col = next((c for c in df.columns if "value" in c or "market" in c), None)

    if ticker_col is None:
        raise ValueError(
            "Simple CSV must have a column named: symbol, ticker, or stock\n"
            f"Found columns: {list(df.columns)}"
        )

    result = pd.DataFrame()
    result["ticker"] = df[ticker_col].str.strip()
    result["quantity"] = pd.to_numeric(df[qty_col], errors="coerce") if qty_col else np.nan
    result["market_value"] = pd.to_numeric(df[mv_col], errors="coerce") if mv_col else np.nan
    result["currency"] = "USD"

    result = result[result["ticker"].notna()]
    print(f"   ✓ Simple CSV format — found {len(result)} positions")
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def fetch_price_data(tickers: list[str], years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)

    print(f"\n📡  Fetching {years}Y price data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    # Drop tickers with too many missing values (>10%)
    threshold = 0.10
    bad = data.columns[data.isna().mean() > threshold].tolist()
    if bad:
        print(f"   ⚠  Dropping tickers with insufficient data: {bad}")
        data = data.drop(columns=bad)

    data = data.ffill().dropna()

    good = data.columns.tolist()
    print(f"   ✓ Loaded data for: {', '.join(good)}")
    return data


# ─────────────────────────────────────────────
# PORTFOLIO MATH
# ─────────────────────────────────────────────
def compute_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple:
    """Return (annual_return, annual_volatility, sharpe_ratio)."""
    port_return = np.dot(weights, mean_returns) * TRADING_DAYS
    port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)
    sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return port_return, port_vol, sharpe


def compute_sortino(weights: np.ndarray, daily_returns: pd.DataFrame) -> float:
    """Compute annualised Sortino ratio using actual downside deviation."""
    port_daily = daily_returns.values @ weights
    port_return = port_daily.mean() * TRADING_DAYS
    daily_rfr = RISK_FREE_RATE / TRADING_DAYS
    downside = port_daily[port_daily < daily_rfr] - daily_rfr
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_dev == 0:
        return 0.0
    return (port_return - RISK_FREE_RATE) / downside_dev


def monte_carlo_portfolios(mean_returns, cov_matrix, n=N_PORTFOLIOS) -> pd.DataFrame:
    """Generate random portfolios via Monte Carlo simulation."""
    n_assets = len(mean_returns)
    results = np.zeros((n, 3 + n_assets))

    for i in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = compute_stats(w, mean_returns, cov_matrix)
        results[i, 0] = ret
        results[i, 1] = vol
        results[i, 2] = sharpe
        results[i, 3:] = w

    cols = ["Return", "Volatility", "Sharpe"] + list(mean_returns.index)
    return pd.DataFrame(results, columns=cols)


def efficient_frontier_curve(mean_returns, cov_matrix, n_points=N_FRONTIER_POINTS) -> pd.DataFrame:
    """Compute the true efficient frontier via sequential optimization."""
    n_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Range of target returns
    min_ret_result = minimize(
        lambda w: compute_stats(w, mean_returns, cov_matrix)[1],
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    min_w = min_ret_result.x
    min_ret = compute_stats(min_w, mean_returns, cov_matrix)[0]
    max_ret = (mean_returns * TRADING_DAYS).max()

    target_returns = np.linspace(min_ret, max_ret * 0.98, n_points)
    frontier = []

    for target in target_returns:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: compute_stats(w, mean_returns, cov_matrix)[0] - t}]
        res = minimize(
            lambda w: compute_stats(w, mean_returns, cov_matrix)[1],
            x0=np.ones(n_assets) / n_assets,
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 1000}
        )
        if res.success:
            ret, vol, sharpe = compute_stats(res.x, mean_returns, cov_matrix)
            frontier.append({"Return": ret, "Volatility": vol, "Sharpe": sharpe, "Weights": res.x})

    return pd.DataFrame(frontier)


def max_sharpe_portfolio(mean_returns, cov_matrix) -> dict:
    """Find the portfolio with maximum Sharpe ratio."""
    n_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    neg_sharpe = lambda w: -compute_stats(w, mean_returns, cov_matrix)[2]
    result = minimize(neg_sharpe, x0=np.ones(n_assets) / n_assets,
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 2000})

    w = result.x
    ret, vol, sharpe = compute_stats(w, mean_returns, cov_matrix)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe}


def min_variance_portfolio(mean_returns, cov_matrix) -> dict:
    """Find the global minimum variance portfolio."""
    n_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: compute_stats(w, mean_returns, cov_matrix)[1],
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    w = result.x
    ret, vol, sharpe = compute_stats(w, mean_returns, cov_matrix)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe}


def max_sortino_portfolio(mean_returns, cov_matrix, daily_returns) -> dict:
    """Find the portfolio with maximum Sortino ratio."""
    n_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: -compute_sortino(w, daily_returns),
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 2000}
    )
    w = result.x
    ret, vol, sharpe = compute_stats(w, mean_returns, cov_matrix)
    sortino = compute_sortino(w, daily_returns)
    return {"weights": w, "return": ret, "volatility": vol, "sharpe": sharpe, "sortino": sortino}


def current_portfolio_stats(holdings: pd.DataFrame, mean_returns, cov_matrix, tickers: list, daily_returns=None) -> dict | None:
    """Compute stats for the actual IBKR portfolio weights."""
    if holdings is None or holdings.empty:
        return None

    holdings = holdings[holdings["ticker"].isin(tickers)].copy()
    if holdings.empty or holdings["market_value"].isna().all():
        return None

    holdings = holdings.dropna(subset=["market_value"])
    holdings = holdings[holdings["market_value"] > 0]
    total_mv = holdings["market_value"].sum()
    holdings["weight"] = holdings["market_value"] / total_mv

    weights = np.zeros(len(tickers))
    for _, row in holdings.iterrows():
        idx = tickers.index(row["ticker"])
        weights[idx] = row["weight"]

    if weights.sum() == 0:
        return None

    weights /= weights.sum()
    ret, vol, sharpe = compute_stats(weights, mean_returns, cov_matrix)
    sortino = compute_sortino(weights, daily_returns) if daily_returns is not None else 0.0
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe, "sortino": sortino}


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_efficient_frontier(
    mc_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    max_sharpe: dict,
    min_var: dict,
    max_sortino: dict,
    current: dict | None,
    tickers: list[str],
    output_path: str = "efficient_frontier.png"
):
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f13")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0f0f13")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    ax.tick_params(colors="#888899", labelsize=9)
    ax.xaxis.label.set_color("#888899")
    ax.yaxis.label.set_color("#888899")

    # Monte Carlo scatter (colored by Sharpe)
    sc = ax.scatter(
        mc_df["Volatility"] * 100, mc_df["Return"] * 100,
        c=mc_df["Sharpe"], cmap="plasma",
        alpha=0.35, s=6, zorder=2
    )

    # Efficient frontier curve
    if not frontier_df.empty:
        ax.plot(
            frontier_df["Volatility"] * 100, frontier_df["Return"] * 100,
            color="#00e5ff", linewidth=2.5, zorder=5, label="Efficient Frontier"
        )

    # Max Sharpe
    ax.scatter(max_sharpe["volatility"] * 100, max_sharpe["return"] * 100,
               s=220, color="#ffdf00", marker="*", zorder=10, label=f"Max Sharpe ({max_sharpe['sharpe']:.2f})")

    # Max Sortino
    ax.scatter(max_sortino["volatility"] * 100, max_sortino["return"] * 100,
               s=180, color="#ff9f43", marker="^", zorder=10, label=f"Max Sortino ({max_sortino['sortino']:.2f})")

    # Min Variance
    ax.scatter(min_var["volatility"] * 100, min_var["return"] * 100,
               s=160, color="#00ff99", marker="D", zorder=10, label=f"Min Variance")

    # Current portfolio
    if current:
        ax.scatter(current["volatility"] * 100, current["return"] * 100,
                   s=200, color="#ff6b6b", marker="P", zorder=10,
                   label=f"Your Portfolio (Sharpe {current['sharpe']:.2f})")
        ax.annotate("  Your\n  Portfolio", (current["volatility"] * 100, current["return"] * 100),
                    color="#ff6b6b", fontsize=8, va="center")

    # Labels for special points
    ax.annotate(f"  Max Sharpe\n  {max_sharpe['return']*100:.1f}% / {max_sharpe['volatility']*100:.1f}% vol",
                (max_sharpe["volatility"] * 100, max_sharpe["return"] * 100),
                color="#ffdf00", fontsize=8, va="bottom")

    ax.annotate(f"  Max Sortino\n  {max_sortino['return']*100:.1f}% / {max_sortino['volatility']*100:.1f}% vol",
                (max_sortino["volatility"] * 100, max_sortino["return"] * 100),
                color="#ff9f43", fontsize=8, va="top")

    ax.annotate(f"  Min Var\n  {min_var['return']*100:.1f}% / {min_var['volatility']*100:.1f}% vol",
                (min_var["volatility"] * 100, min_var["return"] * 100),
                color="#00ff99", fontsize=8, va="top")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Sharpe Ratio", color="#888899", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#888899")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888899", fontsize=8)

    ax.set_xlabel("Annual Volatility (%)", fontsize=11)
    ax.set_ylabel("Annual Return (%)", fontsize=11)
    ax.set_title("Efficient Frontier", color="white", fontsize=18, fontweight="bold", pad=16)

    ax.legend(loc="upper left", facecolor="#1a1a24", edgecolor="#333344",
              labelcolor="white", fontsize=9)
    ax.grid(True, alpha=0.12, color="#334")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    print(f"\n📊  Chart saved → {output_path}")
    plt.show()


def plot_weights(max_sharpe: dict, tickers: list[str], title: str = "Max Sharpe Portfolio Weights",
                 output_path: str = "weights.png"):
    weights = max_sharpe["weights"]
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df = df[df["Weight"] > 0.005].sort_values("Weight", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.55)), facecolor="#0f0f13")
    ax.set_facecolor("#0f0f13")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(df)))
    bars = ax.barh(df["Ticker"], df["Weight"] * 100, color=colors, height=0.6)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row['Weight']*100:.1f}%", va="center", ha="left",
                color="white", fontsize=9)

    ax.set_xlabel("Portfolio Weight (%)", color="#888899")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(True, axis="x", alpha=0.12, color="#334")
    ax.set_xlim(0, df["Weight"].max() * 130)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    print(f"📊  Weights chart saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
def print_report(max_sharpe, min_var, max_sortino, current, tickers, mean_returns, cov_matrix):
    div = "─" * 55

    print(f"\n{'═'*55}")
    print(f"  EFFICIENT FRONTIER ANALYSIS")
    print(f"  Risk-free rate: {RISK_FREE_RATE*100:.1f}%  |  Data window: {LOOKBACK_YEARS}Y")
    print(f"{'═'*55}")

    # Max Sharpe
    print(f"\n  *  MAX SHARPE PORTFOLIO  (Sharpe: {max_sharpe['sharpe']:.3f})")
    print(div)
    print(f"  Expected Return : {max_sharpe['return']*100:.2f}%")
    print(f"  Volatility      : {max_sharpe['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio    : {max_sharpe['sharpe']:.3f}")
    print(f"  Sortino Ratio   : {max_sharpe.get('sortino', 0):.3f}")
    print(f"\n  Weights:")
    w_df = pd.DataFrame({"Ticker": tickers, "Weight": max_sharpe["weights"]})
    w_df = w_df[w_df["Weight"] > 0.005].sort_values("Weight", ascending=False)
    for _, row in w_df.iterrows():
        bar = "█" * int(row["Weight"] * 40)
        print(f"    {row['Ticker']:<8}  {bar:<40}  {row['Weight']*100:5.1f}%")

    # Max Sortino
    print(f"\n  ^  MAX SORTINO PORTFOLIO  (Sortino: {max_sortino['sortino']:.3f})")
    print(div)
    print(f"  Expected Return : {max_sortino['return']*100:.2f}%")
    print(f"  Volatility      : {max_sortino['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio    : {max_sortino['sharpe']:.3f}")
    print(f"  Sortino Ratio   : {max_sortino['sortino']:.3f}")
    print(f"  (Sortino uses downside deviation only — higher = better tail risk profile)")
    w_df3 = pd.DataFrame({"Ticker": tickers, "Weight": max_sortino["weights"]})
    w_df3 = w_df3[w_df3["Weight"] > 0.005].sort_values("Weight", ascending=False)
    print(f"\n  Weights:")
    for _, row in w_df3.iterrows():
        bar = "█" * int(row["Weight"] * 40)
        print(f"    {row['Ticker']:<8}  {bar:<40}  {row['Weight']*100:5.1f}%")

    # Min Variance
    print(f"\n  ◆  MIN VARIANCE PORTFOLIO")
    print(div)
    print(f"  Expected Return : {min_var['return']*100:.2f}%")
    print(f"  Volatility      : {min_var['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio    : {min_var['sharpe']:.3f}")
    w_df2 = pd.DataFrame({"Ticker": tickers, "Weight": min_var["weights"]})
    w_df2 = w_df2[w_df2["Weight"] > 0.005].sort_values("Weight", ascending=False)
    print(f"\n  Weights:")
    for _, row in w_df2.iterrows():
        bar = "█" * int(row["Weight"] * 40)
        print(f"    {row['Ticker']:<8}  {bar:<40}  {row['Weight']*100:5.1f}%")

    # Current portfolio
    if current:
        print(f"\n  ◉  YOUR CURRENT PORTFOLIO")
        print(div)
        print(f"  Expected Return : {current['return']*100:.2f}%")
        print(f"  Volatility      : {current['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio    : {current['sharpe']:.3f}")
        print(f"  Sortino Ratio   : {current.get('sortino', 0):.3f}")
        sharpe_gap = max_sharpe["sharpe"] - current["sharpe"]
        sortino_gap = max_sortino["sortino"] - current.get("sortino", 0)
        if sharpe_gap > 0.1:
            print(f"\n  💡  Optimizing to Max Sharpe could improve Sharpe by +{sharpe_gap:.2f}")
        if sortino_gap > 0.1:
            print(f"  💡  Optimizing to Max Sortino could improve Sortino by +{sortino_gap:.2f}")

    # Correlation matrix hint
    print(f"\n  CORRELATION SUMMARY")
    print(div)
    daily_ret = pd.DataFrame({t: [0] for t in tickers})  # placeholder; real one computed in main
    print(f"  (See correlation heatmap in the saved charts)")

    print(f"\n{'═'*55}\n")


def plot_correlation(price_data: pd.DataFrame, output_path: str = "correlation.png"):
    corr = price_data.pct_change().dropna().corr()

    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.7), max(5, len(corr) * 0.65)),
                           facecolor="#0f0f13")
    ax.set_facecolor("#0f0f13")

    cmap = LinearSegmentedColormap.from_list("rg", ["#d73027", "#ffffbf", "#1a9850"])
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                ax=ax, annot_kws={"size": 8, "color": "black"},
                linewidths=0.5, linecolor="#0f0f13",
                cbar_kws={"shrink": 0.8})

    ax.set_title("Correlation Matrix", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f13")
    print(f"📊  Correlation heatmap saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    global RISK_FREE_RATE

    parser = argparse.ArgumentParser(
        description="Efficient Frontier Calculator with IBKR CSV Import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--ibkr", metavar="FILE", help="Path to IBKR CSV export")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Additional/manual tickers to include (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--years", type=int, default=LOOKBACK_YEARS,
                        help=f"Years of historical data (default: {LOOKBACK_YEARS})")
    parser.add_argument("--rfr", type=float, default=RISK_FREE_RATE,
                        help=f"Risk-free rate, annual (default: {RISK_FREE_RATE})")
    parser.add_argument("--no-plot", action="store_true", help="Skip showing charts (still saves them)")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    args = parser.parse_args()

    RISK_FREE_RATE = args.rfr

    if not args.ibkr and not args.tickers:
        parser.print_help()
        print("\n❌  Provide --ibkr FILE and/or --tickers AAPL MSFT ...\n")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load IBKR holdings ──
    holdings = None
    ibkr_tickers = []
    if args.ibkr:
        holdings = parse_ibkr_csv(args.ibkr)
        ibkr_tickers = holdings["ticker"].dropna().unique().tolist()
        print(f"   Positions: {', '.join(ibkr_tickers)}")

    # ── Combine tickers ──
    extra_tickers = [t.upper() for t in (args.tickers or [])]
    all_tickers = list(dict.fromkeys(ibkr_tickers + extra_tickers))  # preserve order, dedupe

    if not all_tickers:
        print("❌  No tickers found. Check your CSV or provide --tickers.")
        sys.exit(1)

    if len(all_tickers) < 2:
        print("❌  Need at least 2 tickers to compute a frontier.")
        sys.exit(1)

    # ── Fetch price data ──
    prices = fetch_price_data(all_tickers, years=args.years)
    tickers = prices.columns.tolist()  # may have dropped some

    if len(tickers) < 2:
        print("❌  Not enough tickers with valid data.")
        sys.exit(1)

    # ── Compute return stats ──
    daily_returns = prices.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # ── Optimization ──
    print("\n⚙️   Running portfolio optimization...")
    mc_df = monte_carlo_portfolios(mean_returns, cov_matrix)
    frontier_df = efficient_frontier_curve(mean_returns, cov_matrix)
    ms = max_sharpe_portfolio(mean_returns, cov_matrix)
    mv = min_variance_portfolio(mean_returns, cov_matrix)
    mso = max_sortino_portfolio(mean_returns, cov_matrix, daily_returns)

    # Add sortino to max_sharpe dict for reporting
    ms["sortino"] = compute_sortino(ms["weights"], daily_returns)

    # Rename keys for plot compatibility
    ms["return"] = ms.pop("return")
    ms["volatility"] = ms.pop("volatility")
    mv["return"] = mv.pop("return")
    mv["volatility"] = mv.pop("volatility")
    mso["return"] = mso.pop("return")
    mso["volatility"] = mso.pop("volatility")

    current = None
    if holdings is not None:
        current = current_portfolio_stats(holdings, mean_returns, cov_matrix, tickers, daily_returns)
        if current:
            current["return"] = current.pop("return")
            current["volatility"] = current.pop("volatility")

    # ── Individual asset points for context ──
    for ticker in tickers:
        w = np.zeros(len(tickers))
        w[tickers.index(ticker)] = 1.0
        r, v, s = compute_stats(w, mean_returns, cov_matrix)
        # (used in report; plotted via mc)

    # ── Print report ──
    print_report(ms, mv, mso, current, tickers, mean_returns, cov_matrix)

    # ── Charts ──
    if not args.no_plot:
        plot_efficient_frontier(
            mc_df, frontier_df, ms, mv, mso, current, tickers,
            output_path=str(out / "efficient_frontier.png")
        )
        plot_weights(
            ms, tickers, title="Max Sharpe Portfolio — Optimal Weights",
            output_path=str(out / "weights_max_sharpe.png")
        )
        plot_weights(
            mso, tickers, title="Max Sortino Portfolio — Optimal Weights",
            output_path=str(out / "weights_max_sortino.png")
        )
        plot_weights(
            mv, tickers, title="Min Variance Portfolio — Weights",
            output_path=str(out / "weights_min_var.png")
        )
        plot_correlation(prices, output_path=str(out / "correlation.png"))

    # ── Save weights to CSV ──
    weights_out = pd.DataFrame({
        "Ticker": tickers,
        "MaxSharpe_Weight": ms["weights"],
        "MaxSortino_Weight": mso["weights"],
        "MinVar_Weight": mv["weights"],
    })
    if current:
        weights_out["Current_Weight"] = current["weights"]
    weights_path = str(out / "optimal_weights.csv")
    weights_out.to_csv(weights_path, index=False)
    print(f"💾  Weights saved → {weights_path}\n")


if __name__ == "__main__":
    main()
