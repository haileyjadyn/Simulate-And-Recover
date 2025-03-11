# assisted with ChatGPT

import numpy as np
import sys
from ez_diffusion import simulate_and_recover

if __name__ == "__main__":
    N_values = [10, 40, 4000]
    iterations = 1000
    
    for N in N_values:
        print(f"Running simulate-and-recover for N={N}")
        biases, squared_errors = simulate_and_recover(N, iterations)
        
        with open(f"results_N{N}.txt", "w") as f:
            f.write(f"Biases: {biases}\n")
            f.write(f"Squared Errors: {squared_errors}\n")
    
    print("Simulation complete. Results saved.")
