# AI Resume Optimization Engine (Local LLM-Based)

## Overview

The AI Resume Optimization Engine is a lightweight, local AI-powered system designed to automate resume customization based on specific job descriptions.

This project eliminates repetitive manual editing by dynamically aligning resume content with job requirements using a locally hosted LLM.

Built using:
- Ollama (Local LLM runtime)
- Mistral model
- Shell scripting automation
- Structured resume database approach

---

## Problem Statement

Job applications require role-specific resume customization to align with:
- ATS keyword filtering
- Skill prioritization
- Domain-specific language
- Experience relevance

Manual editing for each application is inefficient and error-prone.

---

## Solution Architecture

1. Maintain a structured master resume database
2. Input a job description
3. AI extracts:
   - Required skills
   - Domain keywords
   - Experience signals
4. System filters and rewrites relevant resume sections
5. Outputs an ATS-optimized tailored resume

All processing runs locally — no API cost, no data exposure.

---

## System Workflow

Job Description → Local LLM Processing → Resume Filtering → Keyword Alignment → Final Optimized Resume

---

## Features

- Fully local execution (privacy-first)
- Zero API cost
- ATS-aligned content optimization
- Keyword relevance enhancement
- Structured resume database logic
- Modular and extendable architecture

---

## Installation & Setup

### 1. Install Ollama
Download from: https://ollama.com

### 2. Pull Model
ollama pull mistral

## 3. Run the Script
chmod +x tailor.sh
./tailor.sh

---

## Indian Banking Sector Portfolio Optimizer

This repository also includes a dedicated **41-stock Indian Banking Sector optimizer** (`banking_sector.py`).

### What it does
- Downloads up to 5 years of historical price data for 41 NSE-listed Indian banking stocks via `yfinance`
- Handles missing data and recent IPOs gracefully (forward/back-fill)
- Computes expected returns and the sample covariance matrix using **PyPortfolioOpt**
- Optimizes for **Maximum Sharpe Ratio** (risk-free rate = 6%, proxy for RBI repo rate)
- Prints the cleaned optimal weights to the terminal
- Plots and saves the **Efficient Frontier** (`banking_frontier.png`) with the Max-Sharpe point highlighted
- Plots and saves a **Top-5 Holdings pie chart** (`banking_top5_pie.png`)

### Stocks covered
HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK, BANKBARODA, UNIONBANK, PNB, CANBK, INDIANB, IDBI, AUBANK, YESBANK, FEDERALBNK, INDUSINDBK, IOB, BANKINDIA, IDFCFIRSTB, MAHABANK, BANDHANBNK, UCOBANK, KARURVYSYA, CENTRALBK, RBLBANK, CUB, PSB, J&KBANK, TMB, SOUTHBANK, UJJIVANSFB, KTKBANK, EQUITASBNK, CSBBANK, DCBBANK, UTKARSHBNK, ESAFSFB, DHANBANK, CAPITALSFB, FINOPB *(and JANASFB — delisted)*

### Requirements
```
pip install yfinance pypfopt matplotlib pandas numpy
```

### Run
```
python3 banking_sector.py
```
