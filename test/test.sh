#!/bin/bash
#assisted with AI
# test.sh - Run the test suite for the EZ Diffusion model

echo "Running EZ Diffusion Model Test Suite"
echo "===================================="

# Run the unittest module against our test files
python -m unittest test.test_ez_diffusion
python -m unittest test.test_simulate

echo ""
echo "Test suite complete."