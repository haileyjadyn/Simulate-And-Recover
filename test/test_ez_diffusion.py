import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ez_diffusion import EZDiffusion

class TestEZDiffusion(unittest.TestCase):
    """Test cases for the EZDiffusion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.diffusion = EZDiffusion(T_er=0.3, a=0.1, z=0.5)
        self.nu = 1.2  # Drift rate for testing
        
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test default parameters
        diffusion_default = EZDiffusion()
        self.assertEqual(diffusion_default.T_er, 0.3)
        self.assertEqual(diffusion_default.a, 0.1)
        self.assertEqual(diffusion_default.z, 0.5)
        
        # Test custom parameters
        diffusion_custom = EZDiffusion(T_er=0.5, a=0.2, z=0.7)
        self.assertEqual(diffusion_custom.T_er, 0.5)
        self.assertEqual(diffusion_custom.a, 0.2)
        self.assertEqual(diffusion_custom.z, 0.7)
    
    def test_simulation(self):
        """Test that simulation returns reaction times and choices of the right size."""
        rt, choice = self.diffusion.simulate(self.nu, n_trials=100, seed=42)
        
        # Check that arrays have the right shape
        self.assertEqual(len(rt), 100)
        self.assertEqual(len(choice), 100)
        
        # Check that reaction times are positive
        self.assertTrue(np.all(rt > 0))
        
        # Check that choices are either 0 or 1
        self.assertTrue(np.all((choice == 0) | (choice == 1)))
    
    def test_forward_inverse_consistency(self):
        """Test that applying forward equations followed by inverse equations
        recovers the original drift rate parameter."""
        # Apply forward equations
        mrt, vrt, pc = self.diffusion.forward_equations(self.nu)
        
        # Apply inverse equations
        nu_est = self.diffusion.inverse_equations(mrt, vrt, pc)
        
        # Check that the estimated drift rate is close to the original
        self.assertAlmostEqual(nu_est, self.nu, places=5)
    
    def test_parameter_recovery(self):
        """Test that parameters can be recovered from simulated data."""
        # Simulate data with a very large number of trials for better statistics
        rt, choice = self.diffusion.simulate(self.nu, n_trials=20000, seed=42)
        
        # Compute statistics
        mrt, vrt, pc = self.diffusion.compute_statistics(rt, choice)
        
        # Recover parameters
        nu_est = self.diffusion.inverse_equations(mrt, vrt, pc)
        
        # Check that the estimated drift rate is reasonably close to the original
        # Allow much higher tolerance due to simulation stochasticity
        rel_error = abs(nu_est - self.nu) / self.nu
        self.assertLess(rel_error, 0.2)
    
    def test_parameter_recovery_bias_zero_without_noise(self):
        """Test that the bias is close to zero when there is no noise,
        i.e., when using theoretical statistics."""
        # Use theoretical statistics from forward equations
        mrt, vrt, pc = self.diffusion.forward_equations(self.nu)
        
        # Recover parameters
        nu_est = self.diffusion.inverse_equations(mrt, vrt, pc)
        
        # Compute bias
        bias_nu = nu_est - self.nu
        
        # Check that bias is small
        self.assertAlmostEqual(bias_nu, 0, places=5)
    
    def test_error_handling(self):
        """Test that appropriate errors are raised for invalid inputs."""
        # Test invalid initialization parameters
        with self.assertRaises(ValueError):
            EZDiffusion(T_er=-0.1)  # Negative non-decision time
        
        with self.assertRaises(ValueError):
            EZDiffusion(a=0)  # Zero boundary separation
        
        with self.assertRaises(ValueError):
            EZDiffusion(z=0)  # Zero starting point
        
        with self.assertRaises(ValueError):
            EZDiffusion(z=1)  # Starting point equal to boundary
        
        # Test invalid simulation parameters
        with self.assertRaises(ValueError):
            self.diffusion.simulate(self.nu, n_trials=0)  # Zero trials
        
        # Test invalid inverse equation parameters
        with self.assertRaises(ValueError):
            self.diffusion.inverse_equations(0.5, 0.1, 0)  # Zero proportion correct
        
        with self.assertRaises(ValueError):
            self.diffusion.inverse_equations(0.5, 0.1, 1)  # Perfect accuracy
        
        with self.assertRaises(ValueError):
            self.diffusion.inverse_equations(0.5, 0, 0.7)  # Zero variance
        
        # Test invalid forward equations parameter
        with self.assertRaises(ValueError):
            self.diffusion.forward_equations(0)  # Zero drift rate

if __name__ == "__main__":
    unittest.main()