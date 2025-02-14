import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

st.title("Monte Carlo Simulation for Portfolio Value")

st.write("""
This dashboard simulates your portfolio performance over one year using a Monte Carlo simulation.
You can enter your asset tickers, decide if you want to include a risk‐free mutual fund,
and adjust parameters. The risk assets' parameters (expected return, volatility, correlation)
are estimated from one year of historical data from Yahoo Finance.
""")

# Sidebar Inputs
asset_input = st.sidebar.text_input("Enter asset tickers (comma separated)", 
                                      "ADRO.JK, ITMG.JK, BBCA.JK, BTC-USD, SOL-USD")
# Process tickers; strip extra spaces
risk_assets = [ticker.strip() for ticker in asset_input.split(",") if ticker.strip()]

# Checkbox for mutual fund (risk free asset)
include_mf = st.sidebar.checkbox("Include Mutual Fund (Risk Free Asset)?", value=True)
if include_mf:
    mf_return = st.sidebar.number_input("Mutual Fund Expected Annual Return (decimal)", 
                                          value=0.057, format="%.4f")
    # We'll use a placeholder ticker for mutual fund; it won't fetch data.
    tickers = risk_assets + ["MUTUAL_FUND"]
else:
    tickers = risk_assets.copy()

# Sidebar: Let the user input weights (comma separated, must sum to 1)
default_weights = ",".join(["{:.4f}".format(1/len(tickers)) for _ in tickers])
weights_str = st.sidebar.text_input("Enter weights for each asset (comma separated, sum=1)", 
                                      default_weights)
try:
    weights = np.array([float(x.strip()) for x in weights_str.split(",")])
    if not np.isclose(np.sum(weights), 1):
        st.sidebar.error("Weights must sum to 1.")
except Exception as e:
    st.sidebar.error("Error parsing weights. Using equal weights.")
    weights = np.array([1/len(tickers)] * len(tickers))

# Simulation parameters
num_simulations = st.sidebar.number_input("Number of Simulations", 
                                            min_value=1000, max_value=50000, value=10000, step=1000)
T = 1      # Time horizon (1 year)
N = 252    # Trading days in a year
dt = T / N # Time increment

# Fetch historical data for risk assets using yfinance
# We'll store annualized expected return and volatility for each asset.
asset_mu = []
asset_sigma = []
# For risk assets, also store daily returns for correlation calculation.
returns_dict = {}

for ticker in tickers:
    if ticker == "MUTUAL_FUND":
        # Use user input for mutual fund expected return; assume 0 volatility.
        asset_mu.append(mf_return)
        asset_sigma.append(0.0)
    else:
        try:
            data = yf.download(ticker, period="1y", interval="1d")['Adj Close']
            ret = data.pct_change().dropna()
            returns_dict[ticker] = ret
            daily_mu = ret.mean()
            daily_sigma = ret.std()
            annual_mu = daily_mu * 252
            annual_sigma = daily_sigma * np.sqrt(252)
            asset_mu.append(annual_mu)
            asset_sigma.append(annual_sigma)
        except Exception as e:
            st.error(f"Error fetching data for {ticker}. Using default assumptions.")
            # Default assumptions if data fetch fails
            asset_mu.append(0.10)
            asset_sigma.append(0.35)

asset_mu = np.array(asset_mu)
asset_sigma = np.array(asset_sigma)

# Build the correlation matrix for risk assets.
# For the mutual fund, we set correlation with others to 0.
n = len(tickers)
# Create an empty correlation matrix
corr_matrix = np.eye(n)

# If there are risk assets (non-mutual fund), calculate correlation from historical returns.
risk_tickers = [ticker for ticker in tickers if ticker != "MUTUAL_FUND"]
if len(risk_tickers) > 1:
    returns_df = pd.DataFrame({ticker: returns_dict[ticker] for ticker in risk_tickers})
    corr = returns_df.corr().values
    # Fill the top-left block of the correlation matrix with calculated correlations.
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti != "MUTUAL_FUND" and tj != "MUTUAL_FUND":
                corr_matrix[i, j] = corr[risk_tickers.index(ti), risk_tickers.index(tj)]
# Mutual fund rows and columns remain 0 (except diagonal=1) since it's risk free.

# Compute the covariance matrix
sigma_matrix = np.outer(asset_sigma, asset_sigma)
cov_matrix = sigma_matrix * corr_matrix

st.subheader("Asset Parameters")
for i, ticker in enumerate(tickers):
    st.write(f"**{ticker}**: Annual Expected Return = {asset_mu[i]:.2%}, Annual Volatility = {asset_sigma[i]:.2%}")

# Monte Carlo Simulation
portfolio_final_values = np.zeros(num_simulations)

for i in range(num_simulations):
    portfolio_value = 1.0  # Start with normalized value
    for _ in range(N):
        # Generate daily returns using multivariate normal for all assets
        # For assets with zero variance (e.g., mutual fund), the random draw will return the mean.
        daily_returns = np.random.multivariate_normal(asset_mu * dt, cov_matrix * dt)
        # Alternatively, enforce risk-free asset's daily return:
        for j, ticker in enumerate(tickers):
            if ticker == "MUTUAL_FUND":
                daily_returns[j] = mf_return/252  # deterministic
        port_daily_return = np.dot(weights, daily_returns)
        portfolio_value *= (1 + port_daily_return)
    portfolio_final_values[i] = portfolio_value

# Calculate simulation statistics
expected_value = np.mean(portfolio_final_values)
std_value = np.std(portfolio_final_values)
var_5 = np.percentile(portfolio_final_values, 5)

st.subheader("Simulation Results")
st.write("**Expected Portfolio Value after 1 Year:**", np.round(expected_value, 4))
st.write("**Standard Deviation:**", np.round(std_value, 4))
st.write("**5th Percentile (VaR):**", np.round(var_5, 4))

# Plot the distribution of final portfolio values
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(portfolio_final_values, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
ax.set_xlabel("Portfolio Value after 1 Year")
ax.set_ylabel("Frequency")
ax.set_title("Monte Carlo Simulation of Portfolio Value")
ax.grid(True)
st.pyplot(fig)
