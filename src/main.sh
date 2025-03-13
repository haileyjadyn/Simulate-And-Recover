#!/bin/bash
#assisted with AI
# main.sh - Run the complete EZ Diffusion simulate-and-recover exercise

echo "Starting EZ Diffusion Model Simulate-and-Recover Exercise"
echo "=========================================================="
echo "This will run 1000 iterations for each of the three sample sizes: 10, 40, and 4000"
echo "Total: 3000 iterations"
echo ""

# Create a results directory if it doesn't exist
mkdir -p results

# Run the Python script
python -c "
import numpy as np
from src.simulate import SimulationRunner

# Initialize the simulation runner with 1000 iterations for each sample size
runner = SimulationRunner(n_iterations=1000, sample_sizes=[10, 40, 4000])

# Run the simulations
results = runner.run_simulations()

# Analyze and print the results
summary = runner.analyze_results(results)

# Save results to files
np.save('results/simulation_results.npy', results)
np.save('results/summary_results.npy', summary)

# Generate text files for each sample size
for n in [10, 40, 4000]:
    subset = results[results['sample_size'] == n]
    
    # Calculate biases
    drift_bias = subset['drift_bias'].mean()
    boundary_bias = subset['boundary_bias'].mean()
    nondecision_bias = subset['nondecision_bias'].mean()
    
    # Calculate squared errors
    drift_se = subset['drift_se'].mean()
    boundary_se = subset['boundary_se'].mean() 
    nondecision_se = subset['nondecision_se'].mean()
    
    # Write to file
    with open(f'results_N{n}.txt', 'w') as f:
        f.write(f'N={n}\\n')
        f.write(f'Biases (v, a, t): [{drift_bias:.8f} {boundary_bias:.8f} {nondecision_bias:.8f}]\\n')
        f.write(f'Squared Errors (v, a, t): [{drift_se:.8f} {boundary_se:.8f} {nondecision_se:.8f}]\\n')

print('\\nResults have been saved to the results directory and results_N*.txt files.')
"

echo ""
echo "Simulate-and-recover exercise completed."
echo "See the output above for a summary of the results."
echo "Detailed results have been saved to the results directory."