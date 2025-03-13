#assited with AI

import numpy as np

class EZDiffusion:
    
    def forward_accuracy(self, drift_rate, boundary):
        """Calculate predicted accuracy rate from parameters."""
        y = np.exp(-boundary * drift_rate)
        return 1 / (y + 1)
    
    def forward_mean_rt(self, drift_rate, boundary, nondecision):
        """Calculate predicted mean RT from parameters."""
        y = np.exp(-boundary * drift_rate)
        return nondecision + (boundary / (2 * drift_rate)) * ((1 - y) / (1 + y))
    
    def forward_variance_rt(self, drift_rate, boundary):
        """Calculate predicted RT variance from parameters."""
        y = np.exp(-boundary * drift_rate)
        return ((boundary / (2 * drift_rate))**3) * ((1 - 2*drift_rate*boundary*y - y**2) / ((1 + y)**2))
    
    def inverse_drift_rate(self, accuracy, variance):
        """Calculate drift rate from observed summary statistics."""
        # Handle edge cases
        if accuracy <= 0.5:
            accuracy = 0.501  # Avoid log(0) or negative values
        if accuracy >= 1.0:
            accuracy = 0.999  # Avoid division by zero
            
        L = np.log(accuracy / (1 - accuracy))
        
        # Sign based on whether accuracy is above or below 0.5
        sign = 1 if accuracy > 0.5 else -1
        
        drift_rate = sign * np.sqrt((L**2) / (variance * (L**2 + accuracy * L - accuracy * L**2)))
        
        return drift_rate
    
    def inverse_boundary(self, accuracy, drift_rate):
        """Calculate boundary separation from observed statistics and estimated drift rate."""
        # Handle edge cases
        if accuracy <= 0.5:
            accuracy = 0.501
        if accuracy >= 1.0:
            accuracy = 0.999
            
        L = np.log(accuracy / (1 - accuracy))
        
        return L / drift_rate
    
    def inverse_nondecision(self, mean_rt, drift_rate, boundary):
        """Calculate non-decision time from observed mean RT and estimated parameters."""
        y = np.exp(-boundary * drift_rate)
        
        return mean_rt - (boundary / (2 * drift_rate)) * ((1 - y) / (1 + y))
    
    def recover_parameters(self, accuracy, mean_rt, variance):
        """Recover all parameters from observed summary statistics."""
        drift_rate = self.inverse_drift_rate(accuracy, variance)
        boundary = self.inverse_boundary(accuracy, drift_rate)
        nondecision = self.inverse_nondecision(mean_rt, drift_rate, boundary)
        
        return {
            'drift_rate': drift_rate,
            'boundary': boundary,
            'nondecision': nondecision
        }
    
    def sample_accuracy(self, r_pred, n):
        """Generate a sample accuracy rate from binomial distribution."""
        t_obs = np.random.binomial(n, r_pred)
        return t_obs / n
    
    def sample_mean_rt(self, m_pred, v_pred, n):
        """Generate a sample mean RT from normal distribution."""
        return np.random.normal(m_pred, np.sqrt(v_pred / n))
    
    def sample_variance_rt(self, v_pred, n):
        """Generate a sample variance of RT from gamma distribution."""
        # Gamma parameters
        shape = (n - 1) / 2
        scale = (2 * v_pred) / (n - 1)
        
        return np.random.gamma(shape, scale)
    
    def generate_observed_statistics(self, drift_rate, boundary, nondecision, n):
        """Generate observed summary statistics from parameters."""
        # Calculate predicted summary statistics
        r_pred = self.forward_accuracy(drift_rate, boundary)
        m_pred = self.forward_mean_rt(drift_rate, boundary, nondecision)
        v_pred = self.forward_variance_rt(drift_rate, boundary)
        
        # Generate observed summary statistics with noise
        r_obs = self.sample_accuracy(r_pred, n)
        m_obs = self.sample_mean_rt(m_pred, v_pred, n)
        v_obs = self.sample_variance_rt(v_pred, n)
        
        return r_obs, m_obs, v_obs