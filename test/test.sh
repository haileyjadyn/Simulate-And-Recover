#!/bin/bash
#assited with AI

# test.sh - Run the test suite for the EZ Diffusion model
# This script runs all the unit tests for the EZ Diffusion model implementation

echo "Running EZ Diffusion Model Test Suite"
echo "===================================="

# Run the unittest module against our test files
python -m unittest discover -s test

echo ""
echo "Test suite complete."