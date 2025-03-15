#!/bin/bash
#assisted with AI
# main.sh - Run the complete EZ Diffusion simulate-and-recover exercise

echo "Starting EZ Diffusion Model Simulate-and-Recover Exercise"
echo "=========================================================="
echo "This will run 1000 iterations for each of the three sample sizes: 10, 40, and 4000"
echo "Total: 3000 iterations"
echo ""

# Specify the Python interpreter version
PYTHON_CMD="python3.12"

# Verify Python version
$PYTHON_CMD --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3.12.3 is not available as 'python3.12'. Trying alternative methods..."
    
    # Try python3 with version check
    if python3 --version | grep -q "Python 3.12"; then
        PYTHON_CMD="python3"
        echo "Using $PYTHON_CMD instead."
    else
        echo "Error: Python 3.12.3 is not found. Please install it or update your PATH."
        exit 1
    fi
fi

# Create a results directory if it doesn't exist
mkdir -p results

# Run the Python script
$PYTHON_CMD -c "
import numpy as np
import os
from src.simulate import SimulationRunner

# Check Python version
import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
    print(f'Warning: This script expects Python 3.12 but you are using {sys.version}')

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

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
    
    # Write to file in the results directory
    with open(f'results/results_N{n}.txt', 'w') as f:
        f.write(f'N={n}\\n')
        f.write(f'Biases (v, a, t): [{drift_bias:.8f} {boundary_bias:.8f} {nondecision_bias:.8f}]\\n')
        f.write(f'Squared Errors (v, a, t): [{drift_se:.8f} {boundary_se:.8f} {nondecision_se:.8f}]\\n')

print('\\nResults have been saved to the results directory.')
"

echo ""
echo "Simulate-and-recover exercise completed."
echo "See the output above for a summary of the results."
echo "Detailed results have been saved to the results directory."