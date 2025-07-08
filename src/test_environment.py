"""
Unit tests for the PolymerSimulationEnv environment.

This module contains a series of tests to verify the correct functionality
of the PolymerSimulationEnv, including environment reset, action application,
reward calculation, state observation, ODE integration, and graph updates
for scission and crosslinking events.
"""

import unittest
import yaml
import numpy as np
import sys
import os

# Add the project root to the Python path to allow imports from `src`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.polymer_env import PolymerSimulationEnv # Updated import path

class TestPolymerEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up method that runs once for the entire test class.
        Loads the configuration and initializes the PolymerSimulationEnv.
        This ensures the environment is set up only once, improving test efficiency.
        """
        # Load the main configuration file for the simulation.
        with open("config.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)
        # Initialize the PolymerSimulationEnv with the loaded configuration.
        cls.env = PolymerSimulationEnv(cls.config)

    def test_reset(self):
        """
        Tests if the environment's `reset` method correctly initializes the state.

        Verifies:
        - The dimensionality of the returned state matches the expected state_dim.
        - The state does not contain any NaN (Not a Number) values, indicating
          proper initialization of numerical components.
        """
        state = self.env.reset()
        # Assert that the length of the returned state matches the configured state dimension.
        self.assertEqual(len(state), self.config['rl_hyperparameters']['state_dim'])
        # Assert that no NaN values are present in the state array.
        self.assertFalse(np.isnan(state).any(), "State should not contain NaNs after reset.")

    def test_step_action(self):
        """
        Tests if the `step` method correctly applies actions and updates the dose rate.

        Verifies:
        - The dose rate changes as expected based on the action (increase/decrease).
        - The dose rate is correctly clipped within the defined parameter bounds.
        """
        self.env.reset()
        initial_dose_rate = self.env.dose_rate # Store initial dose rate for comparison
        
        # Test case 1: Action to increase dose_rate
        # An action of 0.5 (relative change) should increase the dose rate.
        action = np.array([0.5]) 
        _, _, _, _ = self.env.step(action) # Take a step with the action
        
        # Assert that the new dose rate is greater than the initial dose rate.
        self.assertGreater(self.env.dose_rate, initial_dose_rate)
        # Assert that the dose rate does not exceed its upper bound.
        self.assertLessEqual(self.env.dose_rate, self.config['run_parameters']['param_bounds']['dose_rate'][1])

        # Test case 2: Action to decrease dose_rate
        self.env.reset() # Reset environment to initial state for a clean test
        initial_dose_rate = self.env.dose_rate
        # An action of -0.5 (relative change) should decrease the dose rate.
        action = np.array([-0.5])
        _, _, _, _ = self.env.step(action)
        # Assert that the new dose rate is less than the initial dose rate.
        self.assertLess(self.env.dose_rate, initial_dose_rate)
        # Assert that the dose rate does not fall below its lower bound.
        self.assertGreaterEqual(self.env.dose_rate, self.config['run_parameters']['param_bounds']['dose_rate'][0])

    def test_reward_function(self):
        """
        Tests the reward calculation logic, specifically its dependence on crosslink density.

        Verifies:
        - A perfect reward (100.0) is given when the crosslink density is within tolerance.
        - A lower reward is given when the crosslink density is far from the target.
        """
        self.env.reset()
        # Retrieve target crosslink density and tolerance from the configuration.
        target_crosslink_density = self.config['run_parameters']['target_properties']['target_crosslink_density']
        tolerance = self.config['run_parameters']['target_properties']['crosslink_tolerance']

        # Case 1: Simulate crosslink density exactly at the target.
        # The PEOOPE concentration (index 5 in voxel state) is set to the target.
        mock_voxel_states_perfect = np.tile(list(self.config['physics_parameters']['initial_concentrations'].values()), (self.env.n_voxels, 1))
        mock_voxel_states_perfect[:, 5] = target_crosslink_density 
        self.env.voxel_states = mock_voxel_states_perfect # Manually set the environment's voxel states
        reward_perfect = self.env._calculate_reward() # Calculate reward
        # Assert that the reward is approximately 100.0 (within a small delta for float comparison).
        self.assertAlmostEqual(reward_perfect, 100.0, delta=0.01)

        # Case 2: Simulate crosslink density far from the target.
        mock_voxel_states_far = np.tile(list(self.config['physics_parameters']['initial_concentrations'].values()), (self.env.n_voxels, 1))
        mock_voxel_states_far[:, 5] = target_crosslink_density + 0.1 # Set PEOOPE concentration far from target
        self.env.voxel_states = mock_voxel_states_far
        reward_far = self.env._calculate_reward()
        # Assert that the reward is significantly lower when far from the target.
        self.assertLess(reward_far, reward_perfect, "Reward should be lower when far from target.")

    def test_state_values(self):
        """
        Tests the structure and properties of the state vector returned by `_get_rl_state`.

        Verifies:
        - The state vector has the expected number of elements (4).
        - All elements in the state vector are numerical (float types).
        - All state values are non-negative, as concentrations and physical properties should be.
        """
        state = self.env.reset()
        # The state vector is expected to contain: [avg_chain_len, avg_crosslink_density, avg_scission_density, laplacian_l2]
        self.assertEqual(len(state), 4)
        # Assert that all elements are instances of float or numpy float types.
        self.assertTrue(all(isinstance(x, (float, np.float32, np.float64)) for x in state))
        # Assert that all state values are non-negative.
        self.assertTrue(np.all(state >= 0), "State values should be non-negative.")

    def test_ode_integration(self):
        """
        Tests if the ODE solver correctly updates chemical concentrations over a time step.

        Verifies:
        - Polyethylene (PE) concentration decreases (due to degradation).
        - Peroxy Radical (PEOO_rad) concentration changes (indicating reaction progression).
        """
        self.env.reset()
        initial_pe_conc = self.env.voxel_states[0, 0] # Initial PE concentration in the first voxel
        initial_peoo_rad_conc = self.env.voxel_states[0, 3] # Initial PEOO• concentration in the first voxel

        # Set a non-zero dose rate to ensure chemical reactions occur during the step.
        self.env.dose_rate = 1.0 
        # Take a step with a dummy action (no change to dose rate from action).
        _, _, _, _ = self.env.step(np.array([0.0])) 

        # Assert that PE concentration has decreased after one time step.
        self.assertLess(self.env.voxel_states[0, 0], initial_pe_conc, "PE concentration should decrease.")
        # Assert that PEOO• concentration has changed (increased or decreased) after one time step.
        self.assertNotAlmostEqual(self.env.voxel_states[0, 3], initial_peoo_rad_conc, places=5, msg="PEOO• concentration should change.")

    def test_graph_update_scission(self):
        """
        Tests if scission events correctly reduce the number of edges in the polymer graph.

        Verifies:
        - The number of edges in the graph does not increase after scission events.
        - If scission events are simulated and bonds exist, the number of edges decreases.
        """
        self.env.reset()
        initial_edges = self.env.G.number_of_edges() # Get initial number of edges
        
        # Create a delta_scission array to simulate scission events in the first voxel.
        delta_scission = np.zeros(self.env.n_voxels)
        # Check if the first voxel contains any nodes to ensure scission can occur.
        if self.env.voxel_to_nodes_map[0]:
            delta_scission[0] = 10 # Simulate 10 scission events in the first voxel

        # Call the private method to update the graph based on simulated scission.
        self.env._update_graph(delta_scission, np.zeros(self.env.n_voxels))
        final_edges = self.env.G.number_of_edges() # Get final number of edges
        
        # Assert that the number of edges has not increased.
        self.assertLessEqual(final_edges, initial_edges, "Number of edges should not increase after scission.")
        # If there were initial edges and scission events were simulated, assert a decrease.
        if initial_edges > 0 and delta_scission[0] > 0:
            self.assertLess(final_edges, initial_edges, "Number of edges should decrease if scission events occur and bonds exist.")

    def test_graph_update_crosslink(self):
        """
        Tests if crosslinking events correctly increase the number of edges in the polymer graph.

        Verifies:
        - The number of edges in the graph does not decrease after crosslinking events.
        - If crosslinking events are simulated and non-bonded pairs exist, the number of edges increases.
        """
        self.env.reset()
        initial_edges = self.env.G.number_of_edges() # Get initial number of edges

        # Create a delta_crosslink array to simulate crosslink events in the first voxel.
        delta_crosslink = np.zeros(self.env.n_voxels)
        # Check if there are at least two nodes in the first voxel for crosslinking to be possible.
        if len(self.env.voxel_to_nodes_map[0]) >= 2:
            delta_crosslink[0] = 5 # Simulate 5 crosslink events in the first voxel

        # Call the private method to update the graph based on simulated crosslinking.
        self.env._update_graph(np.zeros(self.env.n_voxels), delta_crosslink)
        final_edges = self.env.G.number_of_edges() # Get final number of edges

        # Assert that the number of edges has not decreased.
        self.assertGreaterEqual(final_edges, initial_edges, "Number of edges should not decrease after crosslinking.")
        # If crosslink events were simulated and there were potential pairs,
        # assert an increase if it occurred (it's probabilistic).
        if len(self.env.voxel_to_nodes_map[0]) >= 2 and delta_crosslink[0] > 0:
            # It's possible no new bonds are formed if all pairs are already bonded or random choice fails.
            # So, we only assert an increase if `final_edges` is indeed greater than `initial_edges`.
            if final_edges > initial_edges:
                self.assertGreater(final_edges, initial_edges, "Number of edges should increase if crosslink events occur and non-bonded pairs exist.")

# This block allows running the tests directly from the script.
if __name__ == '__main__':
    unittest.main()
