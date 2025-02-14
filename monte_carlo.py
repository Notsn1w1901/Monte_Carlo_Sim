import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

st.title("Monte Carlo Simulation for Portfolio Value")

st.write("""
This dashboard simulates your portfolio performance over one year using a Monte Carlo simulation.
You can enter your asset tickers, decide if you want to include a risk-free mutual fund,
and adjust parameters. The risk assets' parameters (expected return, volatility, correlation)
are estimated from one year of historical data from Yahoo Finance.
""")

# Sidebar Inputs
st.sidebar.image("Designer.png", use_container_width=True)
st.sidebar.header("Portfolio Inputs")

asset_input = st.sidebar.text_input("Enter asset tickers (comma separated)", "AAPL, MSFT, BTC-USD, SOL-USD")
risk_assets = [ticker.strip() for ticker in asset_input.split(",") if ticker.strip()]

# Checkbox for mutual fund (risk-free asset)
include_mf = st.sidebar.checkbox("Include Mutual Fund (Risk Free Asset)?", value=True)
if include_mf:
    mf_return = st.sidebar.number_input("Mutual Fund Expected Annual Return (decimal)", value=0.057, format="%.4f")
    tickers = risk_assets + ["MUTUAL_FUND"]
else:
    tickers = risk_assets.copy()

# User input for Monte Carlo simulation parameters
num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, min_value=100)
N = 252  # Number of trading days in a year
dt = 1 / N  # Time step for simulation

# Initialize lists
asset_mu, asset_sigma = [], []
returns_dict = {}

# Data fetching and processing
for ticker in tickers:
    if ticker == "MUTUAL_FUND":
        asset_mu.append(mf_return)
        asset_sigma.append(0.0)
    else:
        try:
            data = yf.download(ticker, period="1y", interval="1d")
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            ret = prices.pct_change().dropna()
            returns_dict[ticker] = ret
            
            daily_mu = ret.mean()
            daily_sigma = ret.std()
            annual_mu = daily_mu * 252
            annual_sigma = daily_sigma * np.sqrt(252)
            
            asset_mu.append(annual_mu)
            asset_sigma.append(annual_sigma)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}. Using default assumptions.")
            asset_mu.append(0.10)
            asset_sigma.append(0.35)

# Convert lists to NumPy arrays, ensuring they are not empty
if not asset_mu:
    asset_mu = [0.10]
if not asset_sigma:
    asset_sigma = [0.35]

if not asset_mu:
    st.warning("No valid asset data found. Using default return (10%) and volatility (35%).")
    asset_mu = [0.10]  # Default annual return
    asset_sigma = [0.35]  # Default annual volatility

asset_mu = np.array(asset_mu, dtype=float)
asset_sigma = np.array(asset_sigma, dtype=float)

# Build the correlation matrix
n = len(tickers)
corr_matrix = np.eye(n)

risk_tickers = [ticker for ticker in tickers if ticker != "MUTUAL_FUND"]
valid_risk_tickers = [ticker for ticker in risk_tickers if ticker in returns_dict]

if len(valid_risk_tickers) > 1:
    returns_df = pd.DataFrame({ticker: returns_dict[ticker] for ticker in valid_risk_tickers})
    corr = returns_df.corr().values
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti in valid_risk_tickers and tj in valid_risk_tickers:
                idx_i = valid_risk_tickers.index(ti)
                idx_j = valid_risk_tickers.index(tj)
                corr_matrix[i, j] = corr[idx_i, idx_j]

# Compute covariance matrix
sigma_matrix = np.outer(asset_sigma, asset_sigma)
cov_matrix = sigma_matrix * corr_matrix

st.subheader("Asset Parameters")
for i, ticker in enumerate(tickers):
    st.write(f"**{ticker}**: Annual Expected Return = {asset_mu[i]:.2%}, Annual Volatility = {asset_sigma[i]:.2%}")

# Equal-weighted portfolio
weights = np.ones(len(tickers)) / len(tickers)

# Monte Carlo Simulation
simulations = np.zeros((num_simulations, N + 1))
final_values = np.zeros(num_simulations)

for i in range(num_simulations):
    portfolio_value = 1.0
    simulations[i, 0] = portfolio_value
    for t in range(1, N + 1):
        daily_returns = np.random.multivariate_normal(asset_mu * dt, cov_matrix * dt)
        for j, ticker in enumerate(tickers):
            if ticker == "MUTUAL_FUND":
                daily_returns[j] = mf_return / 252
        port_daily_return = np.dot(weights, daily_returns)
        portfolio_value *= (1 + port_daily_return)
        simulations[i, t] = portfolio_value
    final_values[i] = portfolio_value

# Compute statistics
expected_value = np.mean(final_values)
std_value = np.std(final_values)
var_5 = np.percentile(final_values, 5)

st.subheader("Simulation Results")
st.write("**Expected Portfolio Value after 1 Year:**", np.round(expected_value, 4))
st.write("**Standard Deviation:**", np.round(std_value, 4))
st.write("**5th Percentile (VaR):**", np.round(var_5, 4))

# Histogram of final portfolio values
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(final_values, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
ax1.set_xlabel("Portfolio Value after 1 Year")
ax1.set_ylabel("Frequency")
ax1.set_title("Histogram: Monte Carlo Simulation of Final Portfolio Values")
ax1.grid(True)
st.pyplot(fig1)

# Plot simulation paths
num_paths_to_plot = min(100, num_simulations)
sample_indices = np.random.choice(num_simulations, num_paths_to_plot, replace=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
for idx in sample_indices:
    ax2.plot(np.arange(N + 1), simulations[idx, :], alpha=0.5)
ax2.set_xlabel("Trading Days")
ax2.set_ylabel("Portfolio Value (Normalized)")
ax2.set_title("Monte Carlo Simulation Paths")
ax2.grid(True)
st.pyplot(fig2)
