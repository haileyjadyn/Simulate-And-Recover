#assisted with ChatGPT

#!/bin/bash

# Navigate to the src directory
cd "$(dirname "$0")"

echo "Starting simulate-and-recover process..."
python3 simulate.py

echo "Simulation complete. Results saved to results_N10.txt, results_N40.txt, and results_N4000.txt."
