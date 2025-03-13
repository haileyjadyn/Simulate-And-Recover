import numpy as np
from ez_diffusion import EZDiffusion

class SimulationRunner:
    """
    Class to run the simulate-and-recover exercise for the EZ Diffusion model.
    """
    
    def __init__(self, n_iterations=1000, sample_sizes=None):
        """
        Initialize the simulation runner.
        
        Args:
            n_iterations (int): Number of simulation iterations to run per sample size.
            sample_sizes (list): List of sample sizes to test.
        """
        self.n_iterations = n_iterations
        self.sample_sizes = sample_sizes or [10, 40, 4000]
        self.ez_diffusion = EZDiffusion()
        
    def generate_true_parameters(self):
        """
        Randomly select 'true' model parameters within realistic ranges.
        
        Returns:
            tuple: True parameters (nu, alpha, tau)
        """
        # Parameter ranges as specified in the assignment
        alpha = np.random.uniform(0.5, 2.0)  # Boundary separation
        nu = np.random.uniform(0.5, 2.0)     # Drift rate
        tau = np.random.uniform(0.1, 0.5)    # Non-decision time
        
        return nu, alpha, tau
    
    def simulate_and_recover(self, N):
        """
        Run a single simulate-and-recover iteration.
        
        Args:
            N (int): Sample size to use for this iteration.
            
        Returns:
            dict: Results including true parameters, estimated parameters, bias, and squared error.
        """
        # Step 1: Select 'true' parameters
        nu, alpha, tau = self.generate_true_parameters()
        
        # Step 2: Use forward equations to generate 'predicted' summary statistics
        R_pred, M_pred, V_pred = self.ez_diffusion.forward(nu, alpha, tau)
        
        # Step 3: Simulate 'observed' summary statistics
        R_obs, M_obs, V_obs = self.ez_diffusion.generate_observed_statistics(R_pred, M_pred, V_pred, N)
        
        # Step 4: Compute 'estimated' parameters using inverse equations
        try:
            nu_est, alpha_est, tau_est = self.ez_diffusion.inverse(R_obs, M_obs, V_obs)
            
            # Step 5: Compute bias and squared error
            bias_nu = nu - nu_est
            bias_alpha = alpha - alpha_est
            bias_tau = tau - tau_est
            
            se_nu = bias_nu ** 2
            se_alpha = bias_alpha ** 2
            se_tau = bias_tau ** 2
            
            return {
                'true_nu': nu,
                'true_alpha': alpha,
                'true_tau': tau,
                'est_nu': nu_est,
                'est_alpha': alpha_est,
                'est_tau': tau_est,
                'bias_nu': bias_nu,
                'bias_alpha': bias_alpha,
                'bias_tau': bias_tau,
                'se_nu': se_nu,
                'se_alpha': se_alpha,
                'se_tau': se_tau,
                'success': True
            }
            
        except (ValueError, RuntimeWarning, FloatingPointError) as e:
            # Handle potential numerical issues
            return {
                'true_nu': nu,
                'true_alpha': alpha,
                'true_tau': tau,
                'error': str(e),
                'success': False
            }
    
    def run_simulations(self):
        """
        Run multiple simulate-and-recover simulations.
        
        Returns:
            dict: Aggregated results for each sample size.
        """
        results = {}
        
        for N in self.sample_sizes:
            print(f"Running simulations for N = {N}")
            N_results = []
            
            successful_iterations = 0
            for i in range(self.n_iterations):
                if i % 100 == 0 and i > 0:
                    print(f"  Completed {i} iterations...")
                
                iteration_result = self.simulate_and_recover(N)
                N_results.append(iteration_result)
                
                if iteration_result['success']:
                    successful_iterations += 1
            
            print(f"  Completed {self.n_iterations} iterations. Success rate: {successful_iterations/self.n_iterations:.2%}")
            results[N] = N_results
            
        return results
    
    def analyze_results(self, results):
        """
        Analyze simulation results and print summary statistics.
        
        Args:
            results (dict): Simulation results for each sample size.
            
        Returns:
            dict: Summary statistics for each sample size.
        """
        summary = {}
        
        for N, N_results in results.items():
            # Filter out unsuccessful iterations
            successful_results = [r for r in N_results if r['success']]
            n_successful = len(successful_results)
            
            if n_successful == 0:
                print(f"No successful iterations for N = {N}")
                continue
            
            # Calculate mean and standard error of bias and squared error
            bias_nu = np.mean([r['bias_nu'] for r in successful_results])
            bias_alpha = np.mean([r['bias_alpha'] for r in successful_results])
            bias_tau = np.mean([r['bias_tau'] for r in successful_results])
            
            se_nu = np.mean([r['se_nu'] for r in successful_results])
            se_alpha = np.mean([r['se_alpha'] for r in successful_results])
            se_tau = np.mean([r['se_tau'] for r in successful_results])
            
            std_bias_nu = np.std([r['bias_nu'] for r in successful_results]) / np.sqrt(n_successful)
            std_bias_alpha = np.std([r['bias_alpha'] for r in successful_results]) / np.sqrt(n_successful)
            std_bias_tau = np.std([r['bias_tau'] for r in successful_results]) / np.sqrt(n_successful)
            
            print(f"\nResults for N = {N} (successful iterations: {n_successful}):")
            print(f"  Mean bias (nu):    {bias_nu:.6f} ± {std_bias_nu:.6f}")
            print(f"  Mean bias (alpha): {bias_alpha:.6f} ± {std_bias_alpha:.6f}")
            print(f"  Mean bias (tau):   {bias_tau:.6f} ± {std_bias_tau:.6f}")
            print(f"  Mean squared error (nu):    {se_nu:.6f}")
            print(f"  Mean squared error (alpha): {se_alpha:.6f}")
            print(f"  Mean squared error (tau):   {se_tau:.6f}")
            
            summary[N] = {
                'n_successful': n_successful,
                'bias_nu': bias_nu,
                'bias_alpha': bias_alpha,
                'bias_tau': bias_tau,
                'se_nu': se_nu,
                'se_alpha': se_alpha,
                'se_tau': se_tau,
                'std_bias_nu': std_bias_nu,
                'std_bias_alpha': std_bias_alpha,
                'std_bias_tau': std_bias_tau
            }
        
        return summary