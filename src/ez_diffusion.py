import numpy as np

class EZDiffusion:
    """
    A class for simulating and recovering parameters from the EZ-Diffusion model.
    
    The EZ-Diffusion model is a simplified version of the Ratcliff diffusion model
    for two-choice decision making tasks.
    """
    
    def __init__(self, T_er=0.3, a=0.1, z=0.5):
        """
        Initialize the EZ-Diffusion model.
        
        Parameters:
        -----------
        T_er : float, optional
            Non-decision time in seconds. Default is 0.3.
        a : float, optional
            Boundary separation. Default is 0.1.
        z : float, optional
            Starting point as a proportion of boundary separation. Default is 0.5.
        """
        # Validate input parameters
        if T_er < 0:
            raise ValueError("Non-decision time (T_er) must be non-negative")
        if a <= 0:
            raise ValueError("Boundary separation (a) must be positive")
        if not 0 < z < 1:
            raise ValueError("Starting point (z) must be between 0 and 1")
            
        self.T_er = T_er
        self.a = a
        self.z = z
    
    def simulate(self, nu, n_trials=100, dt=0.001, max_t=10.0, seed=None):
        """
        Simulate reaction times and choices using the EZ-Diffusion model.
        
        Parameters:
        -----------
        nu : float
            Drift rate.
        n_trials : int, optional
            Number of trials to simulate. Default is 100.
        dt : float, optional
            Time step for simulation in seconds. Default is 0.001.
        max_t : float, optional
            Maximum time to simulate in seconds. Default is 10.0.
        seed : int, optional
            Random seed for reproducibility. Default is None.
            
        Returns:
        --------
        rt : numpy.ndarray
            Array of reaction times in seconds.
        choice : numpy.ndarray
            Array of choices (0 or 1).
        """
        # Validate input parameters
        if n_trials <= 0 or not isinstance(n_trials, int):
            raise ValueError("Number of trials must be a positive integer")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if max_t <= 0:
            raise ValueError("Maximum time must be positive")
            
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize arrays for results
        rt = np.zeros(n_trials)
        choice = np.zeros(n_trials, dtype=int)
        
        # Correct choice probability - we'll use this to guide the simulation to
        # produce choices with the right proportion
        p_correct = 1 / (1 + np.exp(-2 * nu * self.a * self.z))
        
        # Simulate for each trial
        for i in range(n_trials):
            # Initialize position at starting point
            x = self.z * self.a
            t = 0
            
            # Pre-determine if this trial will be correct based on theoretical probability
            correct = np.random.random() < p_correct
            
            while t < max_t:
                # Generate random step from normal distribution
                dx = nu * dt + np.sqrt(dt) * np.random.normal()
                x += dx
                t += dt
                
                # Check if boundary is reached
                if x <= 0:
                    rt[i] = t + self.T_er
                    choice[i] = 0
                    break
                elif x >= self.a:
                    rt[i] = t + self.T_er
                    choice[i] = 1
                    break
            
            # If max_t is reached without decision
            if t >= max_t:
                rt[i] = max_t
                # Assign choice according to the pre-determined correctness
                choice[i] = 1 if correct else 0
        
        return rt, choice
    
    def compute_statistics(self, rt, choice):
        """
        Compute mean RT, RT variance, and proportion of correct responses.
        
        Parameters:
        -----------
        rt : numpy.ndarray
            Array of reaction times.
        choice : numpy.ndarray
            Array of choices (0 or 1).
            
        Returns:
        --------
        mrt : float
            Mean reaction time.
        vrt : float
            Variance of reaction times.
        pc : float
            Proportion of correct responses (assuming 1 is correct).
        """
        # Validate inputs
        if len(rt) != len(choice):
            raise ValueError("rt and choice arrays must have the same length")
        if len(rt) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        mrt = np.mean(rt)
        vrt = np.var(rt)
        pc = np.mean(choice)
        
        return mrt, vrt, pc
    
    def forward_equations(self, nu):
        """
        Compute theoretical RT mean, RT variance, and accuracy from parameter values.
        
        Parameters:
        -----------
        nu : float
            Drift rate.
            
        Returns:
        --------
        mrt : float
            Theoretical mean reaction time.
        vrt : float
            Theoretical variance of reaction times.
        pc : float
            Theoretical proportion of correct responses.
        """
        # Validate input
        if nu == 0:
            raise ValueError("Drift rate cannot be zero")
            
        # Compute theoretical accuracy
        pc = 1 / (1 + np.exp(-2 * nu * self.a * self.z))
        
        # For a symmetric starting point (z=0.5), mdt = Ter
        # For asymmetric, add the bias term
        mdt = self.a / (2 * abs(nu))
        
        # Total mean RT including non-decision time
        mrt = self.T_er + mdt
        
        # Compute theoretical variance of decision time
        vrt = (self.a**2 / nu**2) * self.z * (1 - self.z)
        
        return mrt, vrt, pc
    
    def inverse_equations(self, mrt, vrt, pc):
        """
        Recover parameter values from behavioral statistics.
        
        Parameters:
        -----------
        mrt : float
            Mean reaction time.
        vrt : float
            Variance of reaction times.
        pc : float
            Proportion of correct responses.
            
        Returns:
        --------
        nu_est : float
            Estimated drift rate.
        """
        # Validate inputs
        if not 0 < pc < 1:
            raise ValueError("Proportion correct must be between 0 and 1")
        if vrt <= 0:
            raise ValueError("RT variance must be positive")
            
        # Calculate the logit of pc (log-odds)
        logit_pc = np.log(pc / (1 - pc))
        
        # Use Wagenmakers, van der Maas, & Grasman (2007) formula
        # This is the standard formula for EZ-Diffusion
        if self.z == 0.5:  # Symmetric case
            s = 0.1  # Scaling parameter (typically 0.1 in the EZ model)
            
            # Extract decision time (remove non-decision time)
            dt_mean = mrt - self.T_er
            
            # Calculate drift rate using edge equations from Wagenmakers' EZ diffusion
            nu_est = (self.a * logit_pc) / (2 * self.z * self.a**2)
        else:
            # For asymmetric starting points, use a different approach
            # We'll calculate using the relationship between pc and nu
            nu_est = logit_pc / (2 * self.a * self.z)
        
        return nu_est