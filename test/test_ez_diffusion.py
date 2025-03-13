#assited with AI

import unittest
import numpy as np
from src.ez_diffusion import EZDiffusion

class TestEZDiffusion(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.ez = EZDiffusion()
        # Define a set of known parameters for testing
        self.test_params = {
            'drift_rate': 1.0,
            'boundary': 1.0,
            'nondecision': 0.3
        }
        
    def test_forward_accuracy(self):
        """Test that the forward equation for accuracy gives expected results"""
        # With drift_rate=1.0 and boundary=1.0, R_pred should be 1/(exp(-1) + 1)
        expected_r = 1 / (np.exp(-1 * self.test_params['drift_rate'] * self.test_params['boundary']) + 1)
        actual_r = self.ez.forward_accuracy(
            self.test_params['drift_rate'], 
            self.test_params['boundary']
        )
        self.assertAlmostEqual(expected_r, actual_r, places=6)
    
    def test_forward_mean_rt(self):
        """Test that the forward equation for mean RT gives expected results"""
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        nondecision = self.test_params['nondecision']
        
        y = np.exp(-drift * boundary)
        expected_mean = nondecision + (boundary / (2 * drift)) * ((1 - y) / (1 + y))
        
        actual_mean = self.ez.forward_mean_rt(drift, boundary, nondecision)
        
        self.assertAlmostEqual(expected_mean, actual_mean, places=6)
    
    def test_forward_variance_rt(self):
        """Test that the forward equation for RT variance gives expected results"""
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        y = np.exp(-drift * boundary)
        expected_var = ((boundary / (2 * drift))**3) * ((1 - 2*drift*boundary*y - y**2) / ((1 + y)**2))
        
        actual_var = self.ez.forward_variance_rt(drift, boundary)
        
        self.assertAlmostEqual(expected_var, actual_var, places=6)
    
    def test_inverse_drift_rate(self):
        """Test inverse equation for drift rate"""
        # Create summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        variance = self.ez.forward_variance_rt(drift, boundary)
        
        # Recover drift rate
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        
        self.assertAlmostEqual(drift, recovered_drift, places=6)
    
    def test_inverse_boundary(self):
        """Test inverse equation for boundary separation"""
        # Create summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        variance = self.ez.forward_variance_rt(drift, boundary)
        
        # Recover drift rate first (needed for boundary calculation)
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        # Recover boundary
        recovered_boundary = self.ez.inverse_boundary(accuracy, recovered_drift)
        
        self.assertAlmostEqual(boundary, recovered_boundary, places=6)
    
    def test_inverse_nondecision(self):
        """Test inverse equation for non-decision time"""
        # Create summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        nondecision = self.test_params['nondecision']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        mean_rt = self.ez.forward_mean_rt(drift, boundary, nondecision)
        variance = self.ez.forward_variance_rt(drift, boundary)
        
        # Recover parameters
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        recovered_boundary = self.ez.inverse_boundary(accuracy, recovered_drift)
        recovered_nondecision = self.ez.inverse_nondecision(mean_rt, recovered_drift, recovered_boundary)
        
        self.assertAlmostEqual(nondecision, recovered_nondecision, places=6)
    
    def test_full_recovery_without_noise(self):
        """Test a full parameter recovery when there's no sampling noise"""
        # Define parameters
        true_params = {
            'drift_rate': 1.5,
            'boundary': 0.8,
            'nondecision': 0.25
        }
        
        # Generate predicted summary statistics
        r_pred = self.ez.forward_accuracy(true_params['drift_rate'], true_params['boundary'])
        m_pred = self.ez.forward_mean_rt(true_params['drift_rate'], true_params['boundary'], true_params['nondecision'])
        v_pred = self.ez.forward_variance_rt(true_params['drift_rate'], true_params['boundary'])
        
        # Set observed = predicted (no noise)
        r_obs, m_obs, v_obs = r_pred, m_pred, v_pred
        
        # Recover parameters
        est_params = self.ez.recover_parameters(r_obs, m_obs, v_obs)
        
        # Check that all parameters are correctly recovered
        self.assertAlmostEqual(true_params['drift_rate'], est_params['drift_rate'], places=6)
        self.assertAlmostEqual(true_params['boundary'], est_params['boundary'], places=6)
        self.assertAlmostEqual(true_params['nondecision'], est_params['nondecision'], places=6)
    
    def test_sampling_distributions(self):
        """Test that sampling distributions generate values with expected properties"""
        # Parameters
        n_samples = 10000
        r_pred = 0.8  # Predicted accuracy rate
        m_pred = 0.5  # Predicted mean RT
        v_pred = 0.1  # Predicted variance of RT
        n = 100       # Sample size
        
        # Generate samples
        r_samples = np.array([self.ez.sample_accuracy(r_pred, n) for _ in range(n_samples)])
        m_samples = np.array([self.ez.sample_mean_rt(m_pred, v_pred, n) for _ in range(n_samples)])
        v_samples = np.array([self.ez.sample_variance_rt(v_pred, n) for _ in range(n_samples)])
        
        # Check that mean of samples is close to predicted value
        self.assertAlmostEqual(r_pred, np.mean(r_samples), places=2)
        self.assertAlmostEqual(m_pred, np.mean(m_samples), places=2)
        self.assertAlmostEqual(v_pred, np.mean(v_samples), places=2)
        
        # Check variances are in expected ranges
        self.assertLess(np.var(r_samples), r_pred * (1 - r_pred) / n + 0.001)  # Binomial variance
        self.assertLess(np.var(m_samples), v_pred / n + 0.001)  # Normal variance

if __name__ == '__main__':
    unittest.main()