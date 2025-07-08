"""
This module defines the PolymerSimulationEnv, a custom OpenAI Gym-like environment
for simulating the degradation of polyethylene polymers. It integrates chemical kinetics
with a graph-based polymer representation to model chain scission and crosslinking.

This environment is designed to be used with reinforcement learning agents,
allowing them to learn optimal strategies for controlling polymer degradation.
"""

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import pandas as pd
import yaml
from pathlib import Path
import random
import json
import sympy

class PolymerSimulationEnv:
    """
    A reinforcement learning environment for simulating polymer degradation based on
    the chemical kinetics model from Sargin & Beckman (2020).

    The environment models the polymer as a collection of voxels, each undergoing
    chemical reactions, and a graph representing the polymer chains.
    """
    def __init__(self, config: dict, scission_model_path: str = None):
        """
        Initializes the Polymer Simulation Environment.

        Args:
            config (dict): A dictionary containing configuration parameters for
                           the simulation, including physics parameters, run parameters,
                           and RL hyperparameters.
            scission_model_path (str, optional): Path to the scission model JSON file.
                                                 If None, defaults to "models/scission_equation.json".
        """
        self.config = config
        self.scission_model_path = scission_model_path
        
        # Load kinetic rate constants from a separate YAML file.
        # This allows for easy modification of reaction rates without changing code.
        kinetic_params_path = Path(self.config['physics_parameters']['kinetic_rate_constants_path'])
        with open(kinetic_params_path, 'r') as f:
            kinetic_config = yaml.safe_load(f)
        # Ensure all kinetic rate constants are floats
        self.k = {key: float(value) for key, value in kinetic_config['rate_constants'].items()} # Store rate constants in self.k

        # --- Load Symbolic Scission Model ---
        self._load_scission_model()

        # --- RL and Simulation Parameters ---
        self.action_space = np.array([-1.0, 1.0], dtype=np.float32)
        self.observation_space = np.array([0.0, 10.0], dtype=np.float32)
        self.param_multipliers = np.array([1.0, 1.0])

        # --- Load Ground-Truth Experimental Data ---
        self._load_experimental_data()

        # Initialize the environment to its starting state.
        self.reset()

    def _load_scission_model(self):
        """Loads the symbolic scission model from the JSON file."""
        path = self.scission_model_path or "models/scission_equation.json"
        try:
            with open(path, 'r') as f:
                scission_model = json.load(f)
            
            sympy_expr = sympy.sympify(scission_model['sympy_expr'])
            # Sort symbols by name to ensure consistent order for lambdify
            self.scission_symbols = tuple(sorted(sympy_expr.free_symbols, key=lambda s: str(s)))
            self.scission_function = sympy.lambdify(self.scission_symbols, sympy_expr, 'numpy')
            print(f"Successfully loaded symbolic scission model from {path}.")
        except FileNotFoundError:
            print(f"Warning: Scission model at '{path}' not found. Using fallback scission model.")
            self.scission_function = None

    def _load_experimental_data(self):
        """Loads the ground-truth experimental data for reward calculation."""
        # Placeholder data with realistic shapes, as used in discover_scission_model.py
        time_hr = np.linspace(0, 2000, 20)
        
        # High dose rate data
        scission_high_dose = 0.05 * (1 - np.exp(-0.002 * time_hr)) + 0.0001 * time_hr
        crosslink_high_dose = 0.001 * time_hr
        
        # Low dose rate data
        scission_low_dose = 0.02 * (1 - np.exp(-0.001 * time_hr)) + 0.00005 * time_hr
        crosslink_low_dose = 0.0005 * time_hr

        self.true_data = {
            10.95: {'time': time_hr, 'scission': scission_high_dose, 'crosslink': crosslink_high_dose},
            0.0108: {'time': time_hr, 'scission': scission_low_dose, 'crosslink': crosslink_low_dose}
        }

    def evaluate_scission_equation(self, dose_rate, t, PEOOH):
        """
        Evaluates the symbolic scission rate equation.
        The symbolic regression was performed on (dose_rate, time) but the model might have
        found a simpler expression. We need to pass the variables it expects.
        """
        if self.scission_function is None:
            # Fallback to original model if symbolic model not loaded
            return self.k['k5'] * PEOOH

        # Based on discover_scission_model.py, PySR will use x0 for dose_rate and x1 for time.
        # We create a map to provide the correct values.
        # The order of self.scission_symbols is guaranteed by sorting them by name in _load_scission_model.
        arg_map = {'x0': dose_rate, 'x1': t}
        
        # Prepare arguments in the correct, sorted order for the lambdified function.
        try:
            args_for_call = [arg_map[str(s)] for s in self.scission_symbols]
        except KeyError as e:
            print(f"ERROR: Symbolic function contains an unknown variable: {e}", file=sys.stderr)
            print(f"Expected variables are 'x0' and 'x1'. Found: {[str(s) for s in self.scission_symbols]}", file=sys.stderr)
            # Return a safe value to prevent crashing the simulation loop.
            return 0.0
        
        return self.scission_function(*args_for_call)


    def _chemical_kinetics_ode(self, t: float, y: list, dose_rate: float, multipliers: np.ndarray) -> list:
        """
        Defines the system of Ordinary Differential Equations (ODEs) for polymer
        degradation kinetics, based on the Sargin & Beckman (2020) model.

        Args:
            t (float): Current time.
            y (list): A list of current concentrations of chemical species.
            dose_rate (float): The current radiation dose rate.
            multipliers (np.ndarray): Array of multipliers for crosslink and scission rates.

        Returns:
            list: A list of the derivatives for each chemical species.
        """
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
        
        # Apply RL-tuned multiplier to the crosslinking rate
        dPEOOPE_dt = (self.k['k4'] * PE_rad**2 + self.k['k6'] * PEOO_rad**2 + \
                       self.k['k7'] * PEOO_rad * PE_rad) * crosslink_multiplier
        
        # Apply RL-tuned multiplier to the symbolic scission rate
        scission_rate = self.evaluate_scission_equation(dose_rate, t, PEOOH)
        dPECOOH_dt = scission_rate * scission_multiplier

        return [dPE_dt, dO2_dt, dPE_rad_dt, dPEOO_rad_dt, dPEOOH_dt, dPEOOPE_dt, dPECOOH_dt]

    def reset(self) -> np.ndarray:
        """
        Resets the environment for a new episode.
        - Resets the parameter multipliers.
        - Returns the initial observation (the multipliers themselves).
        """
        self.param_multipliers = np.array([1.0, 1.0])
        return self._get_rl_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Executes one step of the RL environment.
        1. Applies the action to update the parameter multipliers.
        2. Runs the full ODE simulation for high and low dose rates.
        3. Calculates the reward based on the simulation results.
        4. Returns the new state, reward, and done flag.
        """
        # 1. Apply action to update parameter multipliers
        self.param_multipliers *= np.exp(action * 0.1)
        # Clip multipliers to stay within a reasonable range
        self.param_multipliers = np.clip(self.param_multipliers, 0.1, 10.0)

        # 2. Run full ODE simulation for each dose rate
        predicted_data = {}
        initial_concentrations = list(self.config['physics_parameters']['initial_concentrations'].values())

        for dose_rate, data in self.true_data.items():
            t_span = [data['time'][0], data['time'][-1]]
            t_eval = data['time']
            
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

        # 3. Calculate reward
        reward = self._calculate_reward(predicted_data)
        
        # 4. Return results
        # The episode is always done after one step in this setup.
        done = True
        next_observation = self._get_rl_state()
        
        return next_observation, reward, done, {}

    def _get_rl_state(self) -> np.ndarray:
        """
        Returns the current state for the RL agent, which is the array of
        parameter multipliers.
        """
        return self.param_multipliers

    def _calculate_reward(self, predicted_data: dict) -> float:
        """
        Calculates the reward based on the mean squared error between the
        log of the predicted and true scission/crosslink ratios.
        """
        total_error = 0.0
        num_points = 0

        for dose_rate, pred in predicted_data.items():
            true = self.true_data[dose_rate]
            
            # Avoid division by zero or log(0)
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
        
        # Reward is the negative MSE
        reward = -mean_squared_error
        
        return reward