#assited with AI

import numpy as np
import pandas as pd
import time
from src.ez_diffusion import EZDiffusion

class SimulationResult:
    """Class to store simulation results."""
    def __init__(self, state_history, time_steps):
        self.state_history = state_history
        self.time_steps = time_steps

class Simulator:
    """Class for running general simulations."""
    def __init__(self, dynamics_func=None):
        self.dynamics_func = dynamics_func or self.default_dynamics
    
    def default_dynamics(self, state, dt):
        """Default dynamics function that does a simple increment."""
        return state + dt
    
    def simulate(self, initial_state, num_steps, dt=1.0, noise_level=0.0):
        """Run a simulation for the specified number of steps."""
        # Initialize state history and time steps
        state_history = np.zeros((num_steps + 1, len(initial_state)))
        time_steps = np.zeros(num_steps + 1)
        
        # Set initial state
        state_history[0] = initial_state
        current_state = initial_state.copy()
        
        # Run simulation
        for i in range(1, num_steps + 1):
            # Apply dynamics
            current_state = self.dynamics_func(current_state, dt)
            
            # Add noise if specified
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, size=len(current_state))
                current_state = current_state + noise
            
            # Record state and time
            state_history[i] = current_state
            time_steps[i] = i * dt
        
        return SimulationResult(state_history, time_steps)

class SimulationRunner:
    def __init__(self, n_iterations=1000, sample_sizes=[10, 40, 4000]):
        self.n_iterations = n_iterations
        self.sample_sizes = sample_sizes
        self.ez = EZDiffusion()
        
    def run_simulations(self):
        """Run the simulate-and-recover process for EZ diffusion model."""
        print(f"Running simulate-and-recover process with {self.n_iterations} iterations for each sample size")
        
        # Initialize result storage
        results = []
        
        # Start timer
        start_time = time.time()
        
        # Run simulation for each sample size
        for n in self.sample_sizes:
            print(f"Processing sample size N = {n}")
            
            for i in range(self.n_iterations):
                # Progress indicator every 100 iterations
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Iteration {i + 1}/{self.n_iterations} (Elapsed time: {elapsed:.2f}s)")
                
                # Randomly select parameters
                true_drift = np.random.uniform(0.5, 2.0)
                true_boundary = np.random.uniform(0.5, 2.0)
                true_nondecision = np.random.uniform(0.1, 0.5)
                
                # Generate observed summary statistics
                r_obs, m_obs, v_obs = self.ez.generate_observed_statistics(
                    true_drift, true_boundary, true_nondecision, n
                )
                
                # Recover parameters
                try:
                    est_params = self.ez.recover_parameters(r_obs, m_obs, v_obs)
                    
                    # Calculate bias and squared error
                    drift_bias = true_drift - est_params['drift_rate']
                    boundary_bias = true_boundary - est_params['boundary']
                    nondecision_bias = true_nondecision - est_params['nondecision']
                    
                    drift_se = drift_bias ** 2
                    boundary_se = boundary_bias ** 2
                    nondecision_se = nondecision_bias ** 2
                    
                    # Store results
                    results.append({
                        'sample_size': n,
                        'iteration': i + 1,
                        'true_drift': true_drift,
                        'true_boundary': true_boundary,
                        'true_nondecision': true_nondecision,
                        'est_drift': est_params['drift_rate'],
                        'est_boundary': est_params['boundary'],
                        'est_nondecision': est_params['nondecision'],
                        'drift_bias': drift_bias,
                        'boundary_bias': boundary_bias,
                        'nondecision_bias': nondecision_bias,
                        'drift_se': drift_se,
                        'boundary_se': boundary_se,
                        'nondecision_se': nondecision_se
                    })
                except Exception as e:
                    print(f"Error in iteration {i + 1} with N = {n}: {e}")
                    # Store error case
                    results.append({
                        'sample_size': n,
                        'iteration': i + 1,
                        'true_drift': true_drift,
                        'true_boundary': true_boundary,
                        'true_nondecision': true_nondecision,
                        'est_drift': np.nan,
                        'est_boundary': np.nan,
                        'est_nondecision': np.nan,
                        'drift_bias': np.nan,
                        'boundary_bias': np.nan,
                        'nondecision_bias': np.nan,
                        'drift_se': np.nan,
                        'boundary_se': np.nan,
                        'nondecision_se': np.nan
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
    def analyze_results(self, results_df):
        """Analyze results and generate summary statistics"""
        # Calculate summary statistics
        summary = results_df.groupby('sample_size').agg({
            'drift_bias': ['mean', 'std'],
            'boundary_bias': ['mean', 'std'],
            'nondecision_bias': ['mean', 'std'],
            'drift_se': 'mean',
            'boundary_se': 'mean',
            'nondecision_se': 'mean'
        })
        
        print("\nSummary of Results:")
        print(summary)
        
        return summary

def run_simulation(n_iterations=1000, sample_sizes=[10, 40, 4000]):
    """Run the simulate-and-recover process for EZ diffusion model."""
    # Use the SimulationRunner class
    runner = SimulationRunner(n_iterations=n_iterations, sample_sizes=sample_sizes)
    results = runner.run_simulations()
    summary = runner.analyze_results(results)
    
    # Save results to CSV as well
    results.to_csv('results/results.csv', index=False)
    
    return results, summary

if __name__ == "__main__":
    run_simulation()