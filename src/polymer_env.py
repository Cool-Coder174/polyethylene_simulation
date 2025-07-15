"""
This module defines the PolymerSimulationEnv, a custom Gymnasium environment
for simulating the degradation of polyethylene polymers using LAMMPS.

This environment is designed to be used with reinforcement learning agents,
allowing them to learn optimal strategies for controlling polymer degradation.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import gymnasium as gym
from gymnasium.spaces import Box
import logging
from lammps import lammps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PolymerSimulationEnv(gym.Env):
    """
    A reinforcement learning environment for simulating polymer degradation using LAMMPS.
    """
    def __init__(self, config: dict, experimental_data_path: str = None):
        """
        Initializes the Polymer Simulation Environment.

        Args:
            config (dict): A dictionary containing configuration parameters.
            experimental_data_path (str, optional): Path to the ground-truth experimental data.
                                                    Defaults to "data/experimental_data.json".
        """
        self.config = config
        self.experimental_data_path = experimental_data_path or "data/experimental_data.json"
        
        self._load_experimental_data()

        # Define RL action and observation spaces using Gymnasium's Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.1, high=10.0, shape=(2,), dtype=np.float32)
        
        self.param_multipliers = np.array([1.0, 1.0])
        self.reset()

    def _load_experimental_data(self):
        """Loads the ground-truth experimental data for reward calculation."""
        try:
            with open(self.experimental_data_path, 'r') as f:
                raw_data = json.load(f)
            # Convert lists to numpy arrays for calculations
            self.true_data = {
                float(dose_rate): {key: np.array(val) for key, val in records.items()}
                for dose_rate, records in raw_data.items()
            }
            logging.info(f"Successfully loaded experimental data from {self.experimental_data_path}.")
        except FileNotFoundError:
            logging.error(f"Experimental data file not found: {self.experimental_data_path}")
            raise
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse experimental data file '{self.experimental_data_path}': {e}")
            raise

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.param_multipliers = np.array([1.0, 1.0])
        return self._get_rl_state(), {}

    def step(self, action: np.ndarray):
        """Executes one step of the RL environment."""
        self.param_multipliers *= np.exp(action * 0.1)
        self.param_multipliers = np.clip(self.param_multipliers, self.observation_space.low, self.observation_space.high)

        self.predicted_data = self._run_lammps_simulation()

        reward = self._calculate_reward(self.predicted_data)
        done = True  # Episode is always done after one step
        next_observation = self._get_rl_state()
        
        return next_observation, reward, done, False, {}

    def _run_lammps_simulation(self):
        """Runs the LAMMPS simulation and returns the results."""
        lmp = lammps()
        # TODO: Add LAMMPS commands to run the simulation
        # This will involve creating a LAMMPS input script,
        # running the simulation, and parsing the output.
        # For now, we'll return dummy data.
        return {
            dose_rate: {
                'scission': np.zeros_like(data['time_hr']),
                'crosslink': np.zeros_like(data['time_hr'])
            }
            for dose_rate, data in self.true_data.items()
        }


    def _get_rl_state(self) -> np.ndarray:
        """Returns the current state for the RL agent."""
        return self.param_multipliers.astype(np.float32)

    def _calculate_reward(self, predicted_data: dict) -> float:
        """Calculates reward based on the mean squared error of the log-ratio."""
        total_error = 0.0
        num_points = 0

        for dose_rate, pred in predicted_data.items():
            true = self.true_d ata[dose_rate]
            
            pred_scission = np.maximum(pred['scission'], 1e-9)
            pred_crosslink = np.maximum(pred['crosslink'], 1e-9)
            true_scission = np.maximum(true['scission'], 1e-9)
            true_crosslink = np.maximum(true['crosslink'], 1e-9)

            pred_ratio = pred_scission / pred_crosslink
            true_ratio = true_scission / true_crosslink
            
            log_error = np.log(pred_ratio) - np.log(true_ratio)
            total_error += np.sum(log_error**2)
            num_points += len(pred_ratio)
            
        mean_squared_error = total_error / num_points if num_points > 0 else 0
        return -mean_squared_error

    def get_simulation_data_df(self) -> pd.DataFrame:
        """
        Returns the most recent simulation data as a pandas DataFrame.
        This is intended to be called after a step to log the results.
        """
        data_to_log = []
        if hasattr(self, 'predicted_data'):
            for dose_rate, values in self.predicted_data.items():
                for i, time in enumerate(self.true_data[dose_rate]['time_hr']):
                    data_to_log.append({
                        "run_id": self.run_id,
                        "optuna_trial_id": self.optuna_trial_id,
                        "episode_num": self.episode_num,
                        "day": time,
                        "chain_length_avg": self.chain_length_avg,
                        "num_chains": self.num_chains,
                        "avg_node_connectivity": self.avg_node_connectivity,
                        "crosslinking_pct": values['crosslink'][i],
                        "scission_pct": values['scission'][i],
                        "graph_laplacian_l2": self.graph_laplacian_l2
                    })
        return pd.DataFrame(data_to_log)

    def render(self, mode='human'):
        """Renders the environment (not implemented)."""
        pass

if __name__ == '__main__':
    # Example usage of the PolymerSimulationEnv
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = PolymerSimulationEnv(config)
    obs, _ = env.reset()
    
    # Example of taking a random action
    action = env.action_space.sample()
    next_obs, reward, done, _, _ = env.step(action)
    
    logging.info(f"Initial Observation: {obs}")
    logging.info(f"Action Taken: {action}")
    logging.info(f"Next Observation: {next_obs}")
    logging.info(f"Reward: {reward}")
    logging.info(f"Done: {done}")
    
    # Example of getting the simulation data
    df = env.get_simulation_data_df()
    print(df.head())
