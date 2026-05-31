import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
from adjustText import adjust_text

TICKERS = [
    "HAL.NS", "BEL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BDL.NS",
    "SOLARINDS.NS", "BHARATFORG.NS", "BEML.NS", "GRSE.NS", "MIDHANI.NS",
    "DATAPATTNS.NS", "ZENTEC.NS", "AVANTEL.NS", "APOLLO.NS", "PARAS.NS",
    "MTARTECH.NS", "ASTRAMICRO.NS", "IDEAFORGE.NS",
]

def fetch_prices(tickers, period="5y"):
    print("Fetching historical data from Yahoo Finance...")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    prices = raw.dropna(how="all").ffill()
    print(f"Data shape after cleaning: {prices.shape}")
    return prices

def optimize_portfolio(prices):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
    ef.max_sharpe()
    cleaned = ef.clean_weights()

    print("\n--- Maximum Sharpe Ratio Portfolio Weights ---")
    for ticker, weight in sorted(cleaned.items(), key=lambda x: -x[1]):
        if weight > 1e-4:
            print(f"  {ticker:<18} {weight * 100:.2f}%")

    perf = ef.portfolio_performance(verbose=True)
    return cleaned, perf

def plot_frontier(prices):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(18, 11), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    # --- efficient frontier curve ---
    ef_plot = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False,
                                     ef_param_range=np.linspace(0, 1, 200))
    for line in ax.get_lines():
        line.set_color("#2563eb")
        line.set_linewidth(2.5)

    # --- collect asset positions first so we can set axis limits ---
    assets = []
    for ticker in prices.columns:
        col = prices[ticker].dropna()
        if len(col) < 30:
            continue
        r = mu.get(ticker, np.nan)
        v = np.sqrt(S.loc[ticker, ticker]) if ticker in S.columns else np.nan
        if np.isnan(r) or np.isnan(v):
            continue
        assets.append((ticker.replace(".NS", ""), v, r))

    all_v = [a[1] for a in assets]
    all_r = [a[2] for a in assets]
    x_range = max(all_v) - min(all_v)
    y_range = max(all_r) - min(all_r)
    # generous padding so adjust_text has room to spread labels inside the plot
    ax.set_xlim(min(all_v) - x_range * 0.08, max(all_v) + x_range * 0.55)
    ax.set_ylim(min(all_r) - y_range * 0.30, max(all_r) + y_range * 0.25)

    # --- ticker name labels with a subtle white halo for legibility ---
    texts = []
    for label_str, v, r in assets:
        t = ax.text(
            v, r, label_str,
            fontsize=8, fontweight="bold", color="#1e293b", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
        texts.append(t)

    # pass ax= so adjust_text respects the axes boundaries
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.6),
        expand_text=(1.3, 1.3),
        expand_points=(2.0, 2.0),
        force_text=(0.6, 0.8),
        force_points=(0.5, 0.7),
        lim=500,
    )

    # --- max sharpe star ---
    ef_star = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
    ef_star.max_sharpe()
    perf = ef_star.portfolio_performance()
    star_vol, star_ret = perf[1], perf[0]

    ax.scatter(star_vol, star_ret, marker="*", color="#f59e0b", s=900,
               zorder=5, edgecolors="#d97706", linewidths=1.5)

    ann = ax.annotate(
        "Maximum Sharpe Portfolio",
        xy=(star_vol, star_ret),
        xytext=(star_vol + 0.018, star_ret - 0.025),
        fontsize=11, fontweight="bold", color="#92400e",
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5),
    )
    ann.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # --- styling ---
    ax.set_title("Indian Defense Sector — Efficient Frontier",
                 fontsize=17, fontweight="bold", color="#0f172a", pad=18)
    ax.set_xlabel("Annualised Volatility (Risk)", fontsize=13, color="#334155")
    ax.set_ylabel("Annualised Expected Return", fontsize=13, color="#334155")
    ax.tick_params(colors="#475569", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
    ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.7, alpha=0.9)

    out = "defense_frontier.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nEfficient frontier saved to '{out}'")
    plt.show()

def main():
    prices = fetch_prices(TICKERS)
    optimize_portfolio(prices)
    plot_frontier(prices)

if __name__ == "__main__":
    main()
