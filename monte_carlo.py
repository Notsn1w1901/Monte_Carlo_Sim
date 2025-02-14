import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define assets and updated portfolio weights
assets = ["ADRO.JK", "ITMG.JK", "BBCA.JK", "BTC-USD", "SOL-USD", "Mutual Fund"]
weights = np.array([0.20, 0.20, 0.1676, 0.1324, 0.10, 0.20])

# Assumed annual expected returns (in decimals)
# These are illustrative assumptions:
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

# Compute the covariance matrix from volatilities and correlations
cov_matrix = np.outer(sigma, sigma) * corr_matrix

# Simulation parameters
T = 1            # Time horizon (1 year)
N = 252          # Number of trading days in a year
dt = T / N       # Time increment
num_simulations = 10000  # Number of simulation paths

# Array to store final portfolio values
portfolio_final_values = np.zeros(num_simulations)

# Monte Carlo simulation loop
for i in range(num_simulations):
    portfolio_value = 1.0  # Start with a portfolio value of 1 (or 100%)
    for _ in range(N):
        # Simulate correlated daily returns for all assets
        daily_returns = np.random.multivariate_normal(mu * dt, cov_matrix * dt)
        # Portfolio daily return as weighted sum of asset returns
        port_daily_return = np.dot(weights, daily_returns)
        portfolio_value *= (1 + port_daily_return)
    portfolio_final_values[i] = portfolio_value

# Calculate simulation statistics
expected_value = np.mean(portfolio_final_values)
std_value = np.std(portfolio_final_values)
var_5 = np.percentile(portfolio_final_values, 5)  # 5th percentile (VaR metric)

print("Expected Portfolio Value after 1 Year:", expected_value)
print("Standard Deviation:", std_value)
print("5th Percentile (VaR):", var_5)

# Plot distribution of final portfolio values
plt.figure(figsize=(10,6))
plt.hist(portfolio_final_values, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
plt.xlabel("Portfolio Value after 1 Year")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Portfolio Value")
plt.grid(True)
plt.show()
