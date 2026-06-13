import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import seaborn as sns

# import data, calculate returns, excess returns, expected returns, std, sharpe ratios
closes = yf.download(["^GSPC", "URTH", "BWX", "IAU", "BIL"], period="10y")["Close"]
returns = closes.pct_change().dropna()
excess_returns = returns.subtract(returns["BIL"], axis=0)
exp_excess_returns = excess_returns.mean() * 252
exp_returns = returns.mean() * 252
std_returns = returns.std() * np.sqrt(252)
rf = exp_returns["BIL"]
sharpe_ratios = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


# 2 assets portfolios (using SP500 and Gold, which are the least correlated assets)

# Calculate covariances and variances for excess returns and regular returns
cov_gspc_iau_exc = excess_returns["^GSPC"].cov(excess_returns["IAU"]) * 252
var_gspc_exc = excess_returns["^GSPC"].var() * 252
var_iau_exc = excess_returns["IAU"].var() * 252
cov_gspc_iau = returns["^GSPC"].cov(returns["IAU"]) * 252
# Generate portfolios with different weights and calculate their returns, variances, std, and sharpe ratios
portfolios = []
for x in range(-5, 16):
    w = x / 10
    portfolio_ret = w * exp_returns["^GSPC"] + (1 - w) * exp_returns["IAU"]
    portfolio_var = (w**2) * ((std_returns["^GSPC"])**2) + ((1-w)**2) * ((std_returns["IAU"])**2) + 2 * w * (1 - w) * cov_gspc_iau
    portfolio_std = np.sqrt(portfolio_var)
    port_excess_ret = w * exp_excess_returns["^GSPC"] + (1 - w) * exp_excess_returns["IAU"]
    port_excess_var = (w**2) * var_gspc_exc + ((1-w)**2) * var_iau_exc + 2 * w * (1 - w) * cov_gspc_iau_exc
    portfolio_shp = port_excess_ret / np.sqrt(port_excess_var)
    portfolios.append([w, (1-w), portfolio_ret, portfolio_var, portfolio_std, portfolio_shp])
# store in DataFrame for plotting
two_assets_portfolios = pd.DataFrame(portfolios, columns=["w", "1-w", "portfolio_ret", "portfolio_var", "portfolio_std", "portfolio_shp"])

# min variance portfolio (analytical solution)
var_gspc = std_returns["^GSPC"]**2
var_iau = std_returns["IAU"]**2
w_mvp = (var_iau - cov_gspc_iau) / (var_gspc + var_iau - 2 * cov_gspc_iau)
ret_mvp   = w_mvp * exp_returns["^GSPC"] + (1 - w_mvp) * exp_returns["IAU"]
var_mvp  = w_mvp**2 * var_gspc + (1 - w_mvp)**2 * var_iau + 2 * w_mvp * (1 - w_mvp) * cov_gspc_iau
std_mvp   = np.sqrt(var_mvp)
min_variance_portfolio = pd.Series([w_mvp, 1 - w_mvp, ret_mvp, var_mvp, std_mvp], index=["w", "1-w", "portfolio_ret", "portfolio_var", "portfolio_std"])

# max sharpe portfolio (Montecarlo simulation)
portfolios_mc = []
for i in range(1, 10001):
    w = np.random.uniform(-0.5, 1.5)
    portfolio_ret = w * exp_returns["^GSPC"] + (1 - w) * exp_returns["IAU"]
    portfolio_var = (w**2) * ((std_returns["^GSPC"])**2) + ((1-w)**2) * ((std_returns["IAU"])**2) + 2 * w * (1 - w) * cov_gspc_iau
    portfolio_std = np.sqrt(portfolio_var)
    port_excess_ret = w * exp_excess_returns["^GSPC"] + (1 - w) * exp_excess_returns["IAU"]
    port_excess_var = (w**2) * var_gspc_exc + ((1-w)**2) * var_iau_exc + 2 * w * (1 - w) * cov_gspc_iau_exc
    portfolio_shp = port_excess_ret / np.sqrt(port_excess_var)
    portfolios_mc.append([w, 1-w, portfolio_ret, portfolio_var, portfolio_std, portfolio_shp])
max_sharpe_portfolio = pd.Series(max(portfolios_mc, key=lambda x: x[5]), index=["w", "1-w", "portfolio_ret", "portfolio_var", "portfolio_std", "portfolio_shp"])

# plot efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(two_assets_portfolios["portfolio_std"], two_assets_portfolios["portfolio_ret"], label="Efficient Frontier", color="blue", lw=2)
plt.scatter(min_variance_portfolio["portfolio_std"], min_variance_portfolio["portfolio_ret"], color="red", marker="o", label="Min Variance Portfolio", s=75, edgecolor="black", zorder=5)
plt.scatter(max_sharpe_portfolio["portfolio_std"], max_sharpe_portfolio["portfolio_ret"], color="green", marker="o", label="Max Sharpe Portfolio", s=75, edgecolor="black", zorder=5)
plt.xlabel("Standard Deviation (Annualized)")
plt.ylabel("Expected Return (Annualized)")
plt.title("Efficient Frontier: S&P 500 & Gold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# 5-assets portfolio

# necessary variables for optimization
covar_matrix = returns.cov().values * 252
n = len(exp_returns)
# Define objective functions for optimization
def portfolio_variance(w):
    return w @ covar_matrix @ w
def neg_sharpe(w):
    excess_ret = w @ exp_excess_returns
    portfolio_vol = np.sqrt(w @ covar_matrix @ w)
    return - (excess_ret / portfolio_vol)
def neg_return(w):
    return - (w @ exp_returns)
bounds = [(0.02, 0.3) for _ in range(n)]
constraints_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
def constraints_with_return(r_min):
    return[
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: w @ exp_returns - r_min}
    ]
w0= np.array([1/n] * n)
results = {}

# Perform optimizations for different strategies
results["Min Variance"] = minimize(
    portfolio_variance, w0,
    method="SLSQP", bounds=bounds, constraints=constraints_base)
results["Max Sharpe"] = minimize(
    neg_sharpe, w0,
    method="SLSQP", bounds=bounds,constraints=constraints_base)
results["Max Return"] = minimize(
    neg_return, w0,
    method="SLSQP", bounds=bounds, constraints=constraints_base)
results['Min Var (ret>=6%)'] = minimize(
    portfolio_variance, w0,
    method='SLSQP', bounds=bounds, constraints=constraints_with_return(0.06))
results['Min Var (ret>=8%)'] = minimize(
    portfolio_variance, w0,
    method='SLSQP', bounds=bounds, constraints=constraints_with_return(0.08))

# Analyze results and create summary DataFrame
ticker_to_name = {
    'URTH': 'MSCI World',
    '^GSPC': 'S&P 500',
    'BWX': 'FTSE WGBI',
    'IAU': 'Gold',
    'BIL': 'T-Bill'
}
rows = []
for name, res in results.items():
    if res.success:
        w = res.x
        portfolio_ret = w @ exp_returns
        portfolio_std = np.sqrt(w @ covar_matrix @ w)
        p_daily_excess = returns @ w - returns["BIL"]
        portfolio_shp = (p_daily_excess.mean() / p_daily_excess.std()) * np.sqrt(252)
        row = {ticker_to_name[ticker]: round(v, 4) for ticker, v in zip(returns.columns, w)}
        row['Return (yearly)'] = round(portfolio_ret, 6)
        row['SD (yearly)']     = round(portfolio_std, 6)
        row['Sharpe']          = round(portfolio_shp, 4)
        rows.append(pd.Series(row, name=name))
    else:
        print("Optimization failed")
optimized_portfolios = pd.DataFrame(rows)

# Chart: asset allocation in optimized portfolios and their historical performance (equity curves)
plt.figure(figsize=(10, 6))
optimized_portfolios[list(ticker_to_name.values())].plot(kind='bar', stacked=True, cmap='tab20', ax=plt.gca())
plt.title("Asset Allocation in Optimized Portfolios")
plt.xlabel("Strategy")
plt.ylabel("Weight (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Chart: historical performance of optimized portfolios (equity curves)
plt.figure(figsize=(12, 6))
for name, res in results.items():
    w = res.x
    portfolio_daily_rets = returns @ w
    equity_curve = (1 + portfolio_daily_rets).cumprod() - 1
    plt.plot(equity_curve.index, equity_curve * 100, label=name, lw=2)
plt.title("Historical Performance of Optimized Portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()

# Additional metrics: Beta, Alpha, Treynor Ratio

acwi = yf.download("ACWI", period="10y")["Close"]
acwi_rets = acwi.pct_change().dropna().squeeze()
acwi_rets, rets_aligned = acwi_rets.align(returns, join="inner", axis=0)
exp_returns_aligned = rets_aligned.mean() * 252
rf_aligned = exp_returns_aligned["BIL"]
rm_excess = (acwi_rets - rets_aligned["BIL"]).mean() * 252
portfolio_excess_all = rets_aligned.subtract(rets_aligned["BIL"], axis=0)
acwi_excess = acwi_rets - rets_aligned["BIL"]
acwi_excess_var = acwi_excess.var()
betas = {}
alphas = {}
treynors = {}

# Calculate Beta, Alpha, and Treynor Ratio for each optimized portfolio
for name, res in results.items():
    if res.success:
        w = res.x
        portfolio_excess_rets = portfolio_excess_all @ w
        cov_with_acwi = portfolio_excess_rets.cov(acwi_excess)
        beta = cov_with_acwi / acwi_excess_var
        betas[name] = beta
        rp_excess = np.dot(w, exp_returns_aligned) - rf_aligned
        alphas[name] = rp_excess - (beta * rm_excess)
        treynors[name] = rp_excess / beta if beta != 0 else np.nan

# safe mapping: Aligning directly to the existing DataFrame index
optimized_portfolios["Beta"]     = optimized_portfolios.index.map(betas).round(4)
optimized_portfolios["Alpha"]    = optimized_portfolios.index.map(alphas).round(6)
optimized_portfolios["Treynor"]  = optimized_portfolios.index.map(treynors).round(4)

# sub-periods analysis (Robust Slicing)
returns.index = pd.to_datetime(returns.index)
bear_rets = returns.loc[pd.Timestamp("2021-12-31"):pd.Timestamp("2022-09-30")]
bull_rets = returns.loc[pd.Timestamp("2022-10-31"):pd.Timestamp("2024-12-31")]

# Define functions to calculate max drawdown and max drawup for a return series
def max_drawdown(ret_series):
    cumulative = (1 + ret_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()
def max_drawup(ret_series):
    cumulative = (1 + ret_series).cumprod()
    rolling_min = cumulative.cummin()
    drawup = (cumulative - rolling_min) / rolling_min
    return drawup.max()

# Function to calculate sub-period metrics for a given return series and period type
def subperiod_metrics(ret_series, period, raw_returns_df):
    rf_sub_series = raw_returns_df.loc[ret_series.index, "BIL"]
    excess_ret_series = ret_series - rf_sub_series
    excess_std = excess_ret_series.std()
    sharpe = (excess_ret_series.mean() / excess_std) * np.sqrt(252) if excess_std > 0 else 0.0
    metrics = {
        "Mean return (ann)": ret_series.mean() * 252,
        "Variance (ann)"   : ret_series.var() * 252,
        "SD (ann)"         : ret_series.std() * np.sqrt(252),
        "Sharpe"           : sharpe,
        "Cumulative return": (1 + ret_series).prod() - 1,
    }
    if period == "bear":
        metrics["Max Drawdown"] = max_drawdown(ret_series)
    else:
        metrics["Max Draw-up"] = max_drawup(ret_series)
    return metrics

rows_bear, rows_bull = [], []

# Asset individual metrics
for asset in returns.columns:
    rows_bear.append(pd.Series(subperiod_metrics(bear_rets[asset], "bear", returns), name=asset))
    rows_bull.append(pd.Series(subperiod_metrics(bull_rets[asset], "bull", returns), name=asset))

# Optimized portfolios metrics
for name, res in results.items():
    if res.success:
        w = res.x
        rows_bear.append(pd.Series(subperiod_metrics(bear_rets @ w, "bear", returns), name=name))
        rows_bull.append(pd.Series(subperiod_metrics(bull_rets @ w, "bull", returns), name=name))

df_bear = pd.DataFrame(rows_bear)
df_bull = pd.DataFrame(rows_bull)

# suberiods correlation matrixes heatmaps
corr_bear = bear_rets.corr()
corr_bull = bull_rets.corr()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_bear, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Correlations — Bear market (2022)")
sns.heatmap(corr_bull, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Correlations — Bull market (2022–2024)")
plt.tight_layout()
plt.figure(figsize=(12, 5))

# Drawdown analysis (Underwater Chart) for optimized portfolios
for name, res in results.items():
    w = res.x
    portfolio_daily_rets = returns @ w
    cumulative = (1 + portfolio_daily_rets).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    plt.plot(drawdown.index, drawdown * 100, label=name, alpha=0.7)

plt.title("Analisi Storica dei Drawdown (Underwater Chart)")
plt.xlabel("Data")
plt.ylabel("Distanza dal Massimo (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()