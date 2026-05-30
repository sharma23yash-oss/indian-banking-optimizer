import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting

TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BANKBARODA.NS", "UNIONBANK.NS", "PNB.NS", "CANBK.NS", "INDIANB.NS",
    "IDBI.NS", "AUBANK.NS", "YESBANK.NS", "FEDERALBNK.NS", "INDUSINDBK.NS",
    "IOB.NS", "BANKINDIA.NS", "IDFCFIRSTB.NS", "MAHABANK.NS", "BANDHANBNK.NS",
    "UCOBANK.NS", "KARURVYSYA.NS", "CENTRALBK.NS", "RBLBANK.NS", "CUB.NS",
    "PSB.NS", "J&KBANK.NS", "TMB.NS", "SOUTHBANK.NS", "UJJIVANSFB.NS",
    "KTKBANK.NS", "EQUITASBNK.NS", "CSBBANK.NS", "DCBBANK.NS", "JANASFB.NS",
    "UTKARSHBNK.NS", "ESAFSFB.NS", "DHANBANK.NS", "CAPITALSFB.NS", "FINOPB.NS",
]

# --- 1. Download data (up to 5 years) ---
print("Downloading historical price data for 41 Indian banking stocks...")
raw = yf.download(TICKERS, period="5y", auto_adjust=True, progress=True)["Close"]

# Keep columns that actually came back
raw = raw[[c for c in TICKERS if c in raw.columns]]

# Drop rows where ALL prices are NaN, then forward-fill gaps (recent IPOs)
prices = raw.dropna(how="all").ffill().bfill()

# Drop any ticker that still has no data at all
prices = prices.dropna(axis=1, how="all")

available = list(prices.columns)
print(f"\nTickers with data ({len(available)}): {available}\n")

# --- 2. Expected returns & covariance ---
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# --- 3. Max-Sharpe optimisation (min weight = 0%) ---
ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
ef.max_sharpe(risk_free_rate=0.06)   # ~RBI repo rate proxy
cleaned_weights = ef.clean_weights()

# --- 4. Print weights ---
print("=" * 50)
print("  OPTIMAL PORTFOLIO  (Max Sharpe, min_weight=0%)")
print("=" * 50)
nonzero = {k: v for k, v in cleaned_weights.items() if v > 1e-5}
for ticker, w in sorted(nonzero.items(), key=lambda x: -x[1]):
    print(f"  {ticker:<20s}  {w*100:6.2f}%")
print("-" * 50)
perf = ef.portfolio_performance(verbose=True, risk_free_rate=0.06)
print("=" * 50)

# --- 5 & 6. Efficient Frontier plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the frontier curve
ef_plot = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)

# Scatter individual assets as anonymous dots (no labels)
asset_vols = np.sqrt(np.diag(S.values))
asset_rets = mu.values
ax.scatter(asset_vols, asset_rets, marker="o", s=30, color="steelblue",
           alpha=0.6, zorder=3, label="Individual stocks")

# Mark the Max-Sharpe portfolio
ef_star = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
ef_star.max_sharpe(risk_free_rate=0.06)
ret_star, vol_star, _ = ef_star.portfolio_performance(risk_free_rate=0.06)
ax.scatter(vol_star, ret_star, marker="*", s=300, color="gold",
           edgecolors="darkorange", linewidths=1.2, zorder=5,
           label="Maximum Sharpe Portfolio")
ax.annotate(
    "Maximum Sharpe Portfolio",
    xy=(vol_star, ret_star),
    xytext=(vol_star + 0.005, ret_star + 0.012),
    fontsize=10, fontweight="bold", color="darkorange",
    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
)

ax.set_title("Efficient Frontier — Indian Banking Sector (41 stocks)", fontsize=14)
ax.set_xlabel("Annual Volatility (Std Dev)", fontsize=11)
ax.set_ylabel("Annual Expected Return", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
plt.tight_layout()

output_path = "banking_frontier.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved → {output_path}")

# --- 7. Top-5 holdings pie chart ---
top5 = sorted(nonzero.items(), key=lambda x: -x[1])[:5]
pie_labels = [t for t, _ in top5]
pie_sizes  = [w * 100 for _, w in top5]
pie_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

fig2, ax2 = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax2.pie(
    pie_sizes,
    labels=pie_labels,
    colors=pie_colors,
    explode=[0.04] * len(top5),
    autopct="%1.2f%%",
    pctdistance=0.72,
    startangle=140,
    wedgeprops=dict(linewidth=1.2, edgecolor="white"),
)
for t in texts:
    t.set_fontsize(12)
    t.set_fontweight("bold")
for at in autotexts:
    at.set_fontsize(11)
    at.set_color("white")
    at.set_fontweight("bold")

ax2.set_title("Top 5 Holdings — Max Sharpe Portfolio\nIndian Banking Sector",
              fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()

pie_path = "banking_top5_pie.png"
plt.savefig(pie_path, dpi=150, bbox_inches="tight")
print(f"Pie chart saved → {pie_path}")
