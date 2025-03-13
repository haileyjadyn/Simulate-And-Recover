#!/bin/bash
#assited with AI

# main.sh - Run the complete EZ Diffusion simulate-and-recover exercise
# This script runs the 3000-iteration simulate-and-recovery exercise as required

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

# Save results to a file
np.save('results/simulation_results.npy', results)
np.save('results/summary_results.npy', summary)

print('\nResults have been saved to the results directory.')
"

echo ""
echo "Simulate-and-recover exercise completed."
echo "See the output above for a summary of the results."
echo "Detailed results have been saved to the results directory."