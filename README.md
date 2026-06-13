# Portfolio Optimization with Modern Portfolio Theory

## Overview

This project implements portfolio optimization techniques based on Modern Portfolio Theory (MPT) using historical market data downloaded from Yahoo Finance.

The analysis focuses on constructing efficient portfolios, evaluating their risk-return characteristics, and comparing different investment strategies through several financial performance metrics.

---

## Features

- Historical market data download using `yfinance`
- Daily and annualized return calculation
- Covariance matrix estimation
- Efficient Frontier construction
- Maximum Sharpe Ratio Portfolio optimization
- Minimum Variance Portfolio optimization
- Portfolio allocation under weight constraints
- Monte Carlo portfolio simulation
- Capital Market Line (CML)
- Performance evaluation through:
  - Expected Return
  - Volatility
  - Sharpe Ratio
  - Treynor Ratio
  - Alpha
  - Beta
  - Maximum Drawdown
- Bull and Bear market sub-period analysis
- Graphical visualization of portfolio performance

---

## Technologies

- Python 3
- NumPy
- Pandas
- SciPy
- Matplotlib
- yfinance

---

## Project Structure

```
portfolio_analysis.py
README.md
requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/portfolio-analysis.git
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute

```bash
python portfolio_analysis.py
```

The script automatically:

1. Downloads historical market data
2. Computes daily and annual statistics
3. Generates optimized portfolios
4. Builds the Efficient Frontier
5. Evaluates portfolio performance
6. Produces graphs and summary statistics

---

## Performance Metrics

The project computes several financial indicators, including:

- Expected Annual Return
- Annual Volatility
- Sharpe Ratio
- Treynor Ratio
- Alpha
- Beta
- Maximum Drawdown

---

## Methodology

Portfolio optimization is performed using constrained numerical optimization (`scipy.optimize.minimize`).

The objective functions include:

- Maximum Sharpe Ratio
- Minimum Portfolio Variance

Subject to:

- Portfolio weights sum to one
- Minimum and maximum allocation constraints

---

## Data Source

Historical price data are retrieved from Yahoo Finance using the `yfinance` package.

---

## Educational Purpose

This project was developed as part of an academic study on portfolio management and quantitative finance.
It is intended for educational purposes and should not be interpreted as investment advice.
