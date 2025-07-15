This module defines the PolymerSimulationEnv, a custom Gymnasium environment
for simulating the degradation of polyethylene polymers. It integrates chemical kinetics
with a graph-based polymer representation to model chain scission and crosslinking.

This environment is designed to be used with reinforcement learning agents,
allowing them to learn optimal strategies for controlling polymer degradation.


import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import pandas as pd
import yaml
from pathlib import Path
import random
import json
import sympy
import gymnasium as gym
from gymnasium.spaces import Box
import logging
from lammps import lammps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PolymerSimulationEnv(gym.Env):
    """
    A reinforcement learning environment for simulating polymer degradation based on
    the chemical kinetics model from Sargin & Beckman (2020).

    The environment models the polymer as a collection of voxels, each undergoing
    chemical reactions, and a graph representing the polymer chains.
    """
    def __init__(self, config: dict, scission_model_path: str = None, experimental_data_path: str = None):
        """
        Initializes the Polymer Simulation Environment.

        Args:
            config (dict): A dictionary containing configuration parameters.
            scission_model_path (str, optional): Path to the scission model JSON file.
                                                 Defaults to "models/scission_equation.json".
            experimental_data_path (str, optional): Path to the ground-truth experimental data.
                                                    Defaults to "data/experimental_data.json".
        """
        self.config = config
        self.scission_model_path = scission_model_path or "models/scission_equation.json"
        self.experimental_data_path = experimental_data_path or "data/experimental_data.json"
        
        # Load kinetic rate constants
        kinetic_params_path = Path(self.config['physics_parameters']['kinetic_rate_constants_path'])
        with open(kinetic_params_path, 'r') as f:
            kinetic_config = yaml.safe_load(f)
        self.k = {key: float(value) for key, value in kinetic_config['rate_constants'].items()}

        self._load_scission_model()
        self._load_experimental_data()

        # Define RL action and observation spaces using Gymnasium's Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.1, high=10.0, shape=(2,), dtype=np.float32)
        
        self.param_multipliers = np.array([1.0, 1.0])
        self.reset()

    def _load_scission_model(self):
        """Loads the symbolic scission model from the JSON file."""
        try:
            with open(self.scission_model_path, 'r') as f:
                scission_model = json.load(f)
            
            sympy_expr = sympy.sympify(scission_model['sympy_expr'])
            self.scission_symbols = tuple(sorted(sympy_expr.free_symbols, key=lambda s: str(s)))
            self.scission_function = sympy.lambdify(self.scission_symbols, sympy_expr, 'numpy')
            logging.info(f"Successfully loaded symbolic scission model from {self.scission_model_path}.")
        except FileNotFoundError:
            logging.warning(f"Scission model at '{self.scission_model_path}' not found. Using fallback linear model.")
            self.scission_function = None
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse scission model '{self.scission_model_path}': {e}")
            self.scission_function = None

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

    def evaluate_scission_equation(self, dose_rate, t, PEOOH):
        """Evaluates the symbolic scission rate equation."""
        if self.scission_function is None:
            return self.k['k5'] * PEOOH  # Fallback to original model

        arg_map = {'x0': dose_rate, 'x1': t}
        try:
            args_for_call = [arg_map[str(s)] for s in self.scission_symbols]
        except KeyError as e:
            msg = (f"Symbolic function contains an unknown variable: {e}. "
                   f"Expected 'x0' or 'x1', but symbols are: {[str(s) for s in self.scission_symbols]}")
            logging.error(msg)
            raise ValueError(msg)
        
        return self.scission_function(*args_for_call)

    def _chemical_kinetics_ode(self, t: float, y: list, dose_rate: float, multipliers: np.ndarray) -> list:
        """Defines the system of Ordinary Differential Equations (ODEs) for polymer degradation kinetics."""
        PE, O2, PE_rad, PEOO_rad, PEOOH, PEOOPE, PECOOH = y
        I = dose_rate
        crosslink_multiplier, scission_multiplier = multipliers

        dPE_dt = -2 * self.k['k1'] * PE * I - self.k['k4'] * PEOOH
        dO2_dt = -self.k['k1'] * PE_rad * O2 + self.k['k2'] * PEOO_rad**2 + self.k['k6'] * PEOO_rad**2
        dPE_rad_dt = self.k['k1'] * PE * I - self.k['k2'] * PE_rad * O2 + \
                     self.k['k3'] * PEOO_rad * PE - self.k['k4'] * PE_rad**2 + \
                     self.k['k5'] * PEOOH - self.k['k7'] * PEOO_rad * PE_rad - \
                     self.k['k8'] * PE_rad * PEOOH
        dPEOO_rad_dt = self.k['k1'] * PE_rad * O2 - 2 * self.k['k2'] * PEOO_rad**2 - \
                       self.k['k3'] * PEOO_rad * PE - self.k['k4'] * PEOO_rad - \
                       2 * self.k['k6'] * PEOO_rad**2
        dPEOOH_dt = self.k['k3'] * PEOO_rad * PE - self.k['k5'] * PEOOH - \
                      self.k['k8'] * PE_rad * PEOOH
        
        dPEOOPE_dt = (self.k['k4'] * PE_rad**2 + self.k['k6'] * PEOO_rad**2 + \
                       self.k['k7'] * PEOO_rad * PE_rad) * crosslink_multiplier
        
        scission_rate = self.evaluate_scission_equation(dose_rate, t, PEOOH)
        dPECOOH_dt = scission_rate * scission_multiplier

        return [dPE_dt, dO2_dt, dPE_rad_dt, dPEOO_rad_dt, dPEOOH_dt, dPEOOPE_dt, dPECOOH_dt]

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        self.param_multipliers = np.array([1.0, 1.0])
        return self._get_rl_state(), {}

    def step(self, action: np.ndarray):
        """Executes one step of the RL environment."""
        self.param_multipliers *= np.exp(action * 0.1)
        self.param_multipliers = np.clip(self.param_multipliers, self.observation_space.low, self.observation_space.high)

        if self.config['model_selection']['model'] == 'B':
            self.predicted_data = self._run_ode_simulation()
        elif self.config['model_selection']['model'] == 'C':
            self.predicted_data = self._run_lammps_simulation()
        else:
            raise ValueError(f"Unsupported model type: {self.config['model_selection']['model']}")

        reward = self._calculate_reward(self.predicted_data)
        done = True  # Episode is always done after one step
        next_observation = self._get_rl_state()
        
        return next_observation, reward, done, False, {}

    def _run_ode_simulation(self):
        """Runs the ODE-based simulation."""
        predicted_data = {}
        initial_concentrations = list(self.config['physics_parameters']['initial_concentrations'].values())

        for dose_rate, data in self.true_data.items():
            t_span = [data['time_hr'][0], data['time_hr'][-1]]
            t_eval = data['time_hr']
            
            sol = solve_ivp(
                self._chemical_kinetics_ode,
                t_span,
                initial_concentrations,
                args=(dose_rate, self.param_multipliers),
                method='RK45',
                t_eval=t_eval
            )
            
            predicted_data[dose_rate] = {
                'scission': sol.y[-1],
                'crosslink': sol.y[-2]
            }
        return predicted_data

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
            true = self.true_data[dose_rate]
            
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