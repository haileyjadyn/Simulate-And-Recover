import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
from ez_diffusion import forward_equations, inverse_equations, simulate_data

class TestEZDiffusion(unittest.TestCase):
    def test_forward_inverse_consistency(self):
        """Test that the inverse equations recover the original parameters."""
        v, a, t = 1.0, 1.5, 0.3
        R_pred, M_pred, V_pred = forward_equations(v, a, t)
        v_est, a_est, t_est = inverse_equations(R_pred, M_pred, V_pred)

        print(f"\nExpected v: {v}, Estimated v: {v_est}")
        print(f"Expected a: {a}, Estimated a: {a_est}")
        print(f"Expected t: {t}, Estimated t: {t_est}")

        self.assertAlmostEqual(v, v_est, places=2)
        self.assertAlmostEqual(a, a_est, places=2)
        self.assertAlmostEqual(t, t_est, places=2)

    def test_simulate_data(self):
        """Test that simulated data produces reasonable values."""
        v, a, t, N = 1.0, 1.5, 0.3, 100
        R_obs, M_obs, V_obs = simulate_data(v, a, t, N)
        
        self.assertGreaterEqual(R_obs, 0)
        self.assertLessEqual(R_obs, 1)
        self.assertGreater(M_obs, 0)
        self.assertGreater(V_obs, 0)

if __name__ == "__main__":
    unittest.main()
