#assisted with AI

import unittest
import numpy as np
from simulate import Simulator, SimulationResult

class TestSimulator(unittest.TestCase):
    
    def setUp(self):
        # Create a simulator instance for testing
        self.simulator = Simulator()
    
    def test_simulate_basic_functionality(self):
        """Test that the simulate method returns expected results for a basic scenario."""
        # Define test parameters
        num_steps = 10
        initial_state = np.array([0.5, 0.5])
        
        # Run simulation
        result = self.simulator.simulate(initial_state, num_steps)
        
        # Verify result is a SimulationResult object
        self.assertIsInstance(result, SimulationResult)
        
        # Check that state history has correct shape
        self.assertEqual(result.state_history.shape, (num_steps + 1, 2))
        
        # Verify initial state is correctly recorded
        np.testing.assert_array_equal(result.state_history[0], initial_state)
        
        # Verify time steps are recorded correctly
        self.assertEqual(len(result.time_steps), num_steps + 1)
        self.assertEqual(result.time_steps[0], 0)
        self.assertEqual(result.time_steps[-1], num_steps)
    
    def test_simulate_with_noise(self):
        """Test simulation with noise parameter."""
        # Define test parameters
        initial_state = np.array([0.5, 0.5])
        num_steps = 5
        noise_level = 0.2
        
        # Run simulation without noise first
        result_no_noise = self.simulator.simulate(initial_state, num_steps, noise_level=0.0)
        
        # Run simulation with noise
        result_with_noise = self.simulator.simulate(initial_state, num_steps, noise_level=noise_level)
        
        # Verify result contains the expected number of steps
        self.assertEqual(result_with_noise.state_history.shape, (num_steps + 1, 2))
        
        # Check that noise makes a difference
        # Compare the results without and with noise - they should be different
        # We can't predict exactly how, but we can check they're not identical
        different_values = False
        for i in range(1, num_steps + 1):  # Skip initial state which should be identical
            if not np.array_equal(result_no_noise.state_history[i], result_with_noise.state_history[i]):
                different_values = True
                break
        
        self.assertTrue(different_values, "Noise should cause different simulation results")
    
    def test_simulation_with_custom_dynamics(self):
        """Test simulation with custom dynamics function."""
        # Define a custom dynamics function that doubles the state
        def custom_dynamics(state, dt):
            return state * 2
        
        # Create simulator with custom dynamics
        simulator = Simulator(dynamics_func=custom_dynamics)
        
        # Define test parameters
        initial_state = np.array([1.0, 2.0])
        num_steps = 3
        dt = 0.1
        
        # Run simulation
        result = simulator.simulate(initial_state, num_steps, dt=dt)
        
        # Expected states: [1.0, 2.0] -> [2.0, 4.0] -> [4.0, 8.0] -> [8.0, 16.0]
        expected_final_state = initial_state * (2 ** num_steps)
        np.testing.assert_array_almost_equal(result.state_history[-1], expected_final_state)

if __name__ == '__main__':
    unittest.main()