#assisted with ChatGPT

#!/bin/bash

# Navigate to the test directory
cd "$(dirname "$0")"

echo "Running unit tests for EZ diffusion model..."
python3 -m unittest test_ez_diffusion.py

echo "All tests completed."
