#assisted with ChatGPT

import numpy as np
import scipy.stats as stats

def forward_equations(v, a, t):
    """Compute predicted summary statistics from model parameters."""
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((y + 1) ** 2))
    return R_pred, M_pred, V_pred

def inverse_equations(R_obs, M_obs, V_obs):
    """Estimate model parameters from observed summary statistics."""
    if R_obs <= 0 or R_obs >= 1 or V_obs <= 0:
        return np.nan, np.nan, np.nan
    
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(max(0, L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs))
    a_est = L / v_est if v_est != 0 else np.nan
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))) if v_est != 0 else np.nan
    return v_est, a_est, t_est

def simulate_data(v, a, t, N):
    """Simulate observed data from true model parameters."""
    R_pred, M_pred, V_pred = forward_equations(v, a, t)
    T_obs = np.random.binomial(N, R_pred) if N > 0 else int(R_pred * N)
    R_obs = T_obs / N if N > 0 else 0.5
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N)) if N > 0 else M_pred
    V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred) / (N - 1)) if N > 1 else V_pred
    return R_obs, M_obs, V_obs

def simulate_and_recover(N, iterations=1000):
    """Run simulate-and-recover procedure."""
    biases = []
    squared_errors = []
    
    for _ in range(iterations):
        v = np.random.uniform(0.5, 2)
        a = np.random.uniform(0.5, 2)
        t = np.random.uniform(0.1, 0.5)
        
        R_obs, M_obs, V_obs = simulate_data(v, a, t, N)
        v_est, a_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
        
        if not np.isnan(v_est) and not np.isnan(a_est) and not np.isnan(t_est):
            bias = np.array([v - v_est, a - a_est, t - t_est])
            squared_error = bias ** 2
            biases.append(bias)
            squared_errors.append(squared_error)
    
    biases = np.mean(biases, axis=0) if biases else np.array([np.nan, np.nan, np.nan])
    squared_errors = np.mean(squared_errors, axis=0) if squared_errors else np.array([np.nan, np.nan, np.nan])
    return biases, squared_errors
