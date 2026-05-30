# 🏦 Indian Banking Sector Quantitative Optimizer

## Project Overview

A specialized financial modeling tool that applies Modern Portfolio Theory (MPT) to the entire Indian Banking sector. This algorithm analyzes 41 major public, private, and small finance banks to construct a mathematically optimized, high-yield portfolio.

Unlike standard index funds, this model utilizes a 0% minimum allocation floor, allowing the algorithm to ruthlessly eliminate underperforming or highly correlated assets. The result is a hyper-focused portfolio designed to strictly maximize the Sharpe Ratio.

---

## ⚙️ Technical Stack & Methodology

- **Data Extraction:** Up to 5 years of historical daily closing prices fetched via the `yfinance` API.
- **Optimization Engine:** PyPortfolioOpt calculates annualized expected returns and the sample covariance matrix.
- **Strategy:** Maximum Sharpe Ratio (Risk-Adjusted Return).
- **Output:** Generates an Efficient Frontier scatter plot and isolates the exact percentage weights of the winning assets.

---

## 📊 The 41-Bank Universe

Includes heavyweights like HDFC Bank, ICICI Bank, and SBI, along with mid-caps and small finance banks like Federal Bank, Karur Vysya, and Equitas.

*(Note: The algorithm successfully distilled these 41 down to the top 5 pure risk-adjusted performers).*

---

## 🚀 How to Run Locally

1. Install dependencies:
```
pip install yfinance PyPortfolioOpt matplotlib pandas
```

2. Run the script:
```
python banking_sector.py
```
