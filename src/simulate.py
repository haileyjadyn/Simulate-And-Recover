#assited with AI

import numpy as np
import pandas as pd
import time
from src.ez_diffusion import EZDiffusion

def run_simulation(n_iterations=1000, sample_sizes=[10, 40, 4000]):
    """Run the simulate-and-recover process for EZ diffusion model."""
    print(f"Running simulate-and-recover process with {n_iterations} iterations for each sample size")
    
    # Initialize EZ diffusion model
    ez = EZDiffusion()
    
    # Initialize result storage
    results = []
    
    # Start timer
    start_time = time.time()
    
    # Run simulation for each sample size
    for n in sample_sizes:
        print(f"Processing sample size N = {n}")
        
        for i in range(n_iterations):
            # Progress indicator every 100 iterations
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Iteration {i + 1}/{n_iterations} (Elapsed time: {elapsed:.2f}s)")
            
            # Randomly select parameters
            true_drift = np.random.uniform(0.5, 2.0)
            true_boundary = np.random.uniform(0.5, 2.0)
            true_nondecision = np.random.uniform(0.1, 0.5)
            
            # Generate observed summary statistics
            r_obs, m_obs, v_obs = ez.generate_observed_statistics(
                true_drift, true_boundary, true_nondecision, n
            )
            
            # Recover parameters
            try:
                est_params = ez.recover_parameters(r_obs, m_obs, v_obs)
                
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
    
    # Save results to CSV
    results_df.to_csv('results.csv', index=False)
    summary.to_csv('summary.csv')
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds")
    print("Results saved to 'results.csv' and 'summary.csv'")
    
    return results_df

if __name__ == "__main__":
    run_simulation()