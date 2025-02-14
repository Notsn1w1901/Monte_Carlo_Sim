import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Monte Carlo Simulation for Portfolio Value")
st.write("""
This dashboard simulates the portfolio performance over one year using a Monte Carlo simulation.
The portfolio consists of:
- **ADRO.JK**
- **ITMG.JK**
- **BBCA.JK**
- **BTC-USD**
- **SOL-USD**
- **Mutual Fund** (Money Market, ~5.7% expected return)
""")

# Sidebar for simulation parameters
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
T = 1            # Time horizon in years
N = 252          # Trading days in a year
dt = T / N       # Time increment

# Define assets and portfolio weights
assets = ["ADRO.JK", "ITMG.JK", "BBCA.JK", "BTC-USD", "SOL-USD", "Mutual Fund"]
weights = np.array([0.20, 0.20, 0.1676, 0.1324, 0.10, 0.20])

# Assumed annual expected returns (in decimals)
mu = np.array([0.10, 0.12, 0.08, 0.30, 0.25, 0.057])

# Assumed annual volatilities (standard deviation)
sigma = np.array([0.35, 0.40, 0.30, 0.90, 1.00, 0.02])

# Define a simplified correlation matrix among assets
corr_matrix = np.array([
    [1.0, 0.6, 0.5, 0.1, 0.1, 0.0],
    [0.6, 1.0, 0.5, 0.1, 0.1, 0.0],
    [0.5, 0.5, 1.0, 0.1, 0.1, 0.0],
    [0.1, 0.1, 0.1, 1.0, 0.2, 0.0],
    [0.1, 0.1, 0.1, 0.2, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])

# Compute the covariance matrix
cov_matrix = np.outer(sigma, sigma) * corr_matrix

# Monte Carlo simulation
portfolio_final_values = np.zeros(num_simulations)

for i in range(num_simulations):
    portfolio_value = 1.0  # Start with a normalized portfolio value (100%)
    for _ in range(N):
        # Generate daily returns from a multivariate normal distribution
        daily_returns = np.random.multivariate_normal(mu * dt, cov_matrix * dt)
        port_daily_return = np.dot(weights, daily_returns)
        portfolio_value *= (1 + port_daily_return)
    portfolio_final_values[i] = portfolio_value

# Calculate simulation statistics
expected_value = np.mean(portfolio_final_values)
std_value = np.std(portfolio_final_values)
var_5 = np.percentile(portfolio_final_values, 5)  # 5th percentile as VaR

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
