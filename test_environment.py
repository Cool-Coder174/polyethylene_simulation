# test_environment.py
import unittest
import yaml
import numpy as np
from polymer_env import PolymerSimulationEnv

class TestPolymerEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load config and initialize the environment once for all tests."""
        with open("config.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)
        cls.env = PolymerSimulationEnv(cls.config)

    def test_reset(self):
        """Test if the environment resets to the correct initial state."""
        state = self.env.reset()
        self.assertEqual(len(state), self.config['agent']['state_dim'])
        self.assertFalse(np.isnan(state).any(), "State should not contain NaNs after reset.")

    def test_step_action(self):
        """Test if a step action correctly modifies the parameters within bounds."""
        self.env.reset()
        initial_lchain = self.env.params['Lchain']
        
        # Action that should increase Lchain
        action = np.array([0.5, 0, 0, 0]) # +10% Lchain if max_action is 0.2
        self.env.step(action)
        
        self.assertGreater(self.env.params['Lchain'], initial_lchain)
        # Check if it's clipped at the upper bound
        self.assertLessEqual(self.env.params['Lchain'], self.config['simulation']['param_bounds']['Lchain'][1])

    def test_reward_function(self):
        """Test the goal-oriented reward logic."""
        # --- Case 1: Connectivity is exactly at the target ---
        target_conn = self.config['simulation']['reward_targets']['target_connectivity']
        
        # Mock a dataframe where the final connectivity is on target
        mock_df_perfect = pd.DataFrame({'avg_node_connectivity': [target_conn], 'graph_laplacian_l2': [0.05]})
        reward_perfect, _ = self.env._calculate_reward_and_next_state(mock_df_perfect, action=np.zeros(4), sim_duration=10)
        
        # The quality reward should be at its maximum (100)
        self.assertAlmostEqual(reward_perfect, 100.0 - (0.1*10), delta=0.1)

        # --- Case 2: Connectivity is far from the target ---
        mock_df_far = pd.DataFrame({'avg_node_connectivity': [target_conn + 2], 'graph_laplacian_l2': [0.02]})
        reward_far, _ = self.env._calculate_reward_and_next_state(mock_df_far, action=np.zeros(4), sim_duration=10)

        self.assertLess(reward_far, reward_perfect, "Reward should be lower when far from target.")

    def test_state_normalization(self):
        """Ensure the state vector is properly normalized."""
        state = self.env.reset()
        # All values in the normalized state should be between 0 and ~1
        self.assertTrue(np.all(state >= 0) and np.all(state <= 1.5), "Normalized state is out of expected [0, 1.5] range.")

if __name__ == '__main__':
    unittest.main()

