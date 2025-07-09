"""
Tests for the PolymerSimulationEnv, including the hybrid symbolic-RL modeling components.
"""
import pytest
import numpy as np
import json
import yaml
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.polymer_env import PolymerSimulationEnv

@pytest.fixture(scope="module")
def config():
    """Fixture to load the main configuration for all tests in this module."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def dummy_scission_model_path(tmp_path):
    """
    Fixture to create a dummy symbolic scission model file in a temporary directory.
    This provides a consistent, simple model for testing purposes.
    """
    equation_file = tmp_path / "scission_equation.json"
    # A simple linear equation: 0.1 * x0 + 0.01 * x1 (representing dose_rate and time)
    equation_details = {
        "latex": "0.1 x_{0} + 0.01 x_{1}",
        "sympy_expr": "0.1*x0 + 0.01*x1",
    }
    with open(equation_file, 'w') as f:
        json.dump(equation_details, f)
    return str(equation_file)

@pytest.fixture
def env(config, dummy_scission_model_path):
    """Fixture to create a fresh instance of the PolymerSimulationEnv for each test."""
    return PolymerSimulationEnv(config, scission_model_path=dummy_scission_model_path)

def test_environment_initialization(env, config):
    """
    Tests if the environment initializes correctly with the given configuration.
    """
    assert env.config == config
    assert env.scission_function is not None, "Symbolic scission function should be loaded."
    assert np.array_equal(env.param_multipliers, [1.0, 1.0]), "Initial multipliers should be [1.0, 1.0]."

def test_reset_method(env):
    """
    Tests if the reset method correctly resets the parameter multipliers and returns the initial state.
    """
    # Modify the state, then reset
    env.param_multipliers = np.array([2.0, 3.0])
    initial_state = env.reset()
    
    assert np.array_equal(initial_state, [1.0, 1.0]), "Reset should return the initial state [1.0, 1.0]."
    assert np.array_equal(env.param_multipliers, [1.0, 1.0]), "Multipliers should be reset to [1.0, 1.0]."

def test_scission_equation_loading_and_evaluation(env):
    """
    Tests that the environment can correctly load and evaluate the symbolic scission equation.
    """
    # Test evaluation with known values
    dose_rate = 10.0
    time = 100.0
    # Expected from dummy model: 0.1 * 10.0 + 0.01 * 100.0 = 1.0 + 1.0 = 2.0
    expected_rate = 2.0
    
    # The 'evaluate_scission_equation' method expects PEOOH, but the dummy model doesn't use it.
    # We pass a dummy value for it.
    rate = env.evaluate_scission_equation(dose_rate, time, PEOOH=0)
    
    assert np.isclose(rate, expected_rate, atol=1e-5)

def test_step_method_and_action_effect(env):
    """
    Tests that the step method correctly applies an action, updates multipliers,
    and returns a valid state, reward, and done flag.
    """
    env.reset()
    
    # Action should be a small multiplicative change
    action = np.array([0.1, -0.1])
    next_state, reward, done, info = env.step(action)
    
    # Check if multipliers were updated correctly
    # np.exp(0.1 * 0.1) approx 1.01
    # np.exp(-0.1 * 0.1) approx 0.99
    assert np.isclose(env.param_multipliers[0], 1.0 * np.exp(0.1 * 0.1))
    assert np.isclose(env.param_multipliers[1], 1.0 * np.exp(-0.1 * 0.1))
    
    # Check return types and values
    assert isinstance(reward, float) and np.isfinite(reward), "Reward should be a finite float."
    assert done is True, "Episode should be done after one step."
    assert isinstance(info, dict), "Info should be a dictionary."
    assert np.array_equal(next_state, env.param_multipliers), "Next state should be the new multipliers."

def test_reward_calculation(env):
    """
    Tests the reward calculation with a known set of predicted and true data.
    """
    # Mock predicted data
    predicted_data = {
        10.95: {'scission': np.array([0.1, 0.2]), 'crosslink': np.array([0.05, 0.1])},
        0.0108: {'scission': np.array([0.05, 0.1]), 'crosslink': np.array([0.02, 0.05])}
    }
    
    # True ratios from the mocked data in the environment
    # High dose: scission/crosslink -> [2.0, 2.0]
    # Low dose: scission/crosslink -> [2.5, 2.0]
    
    # Predicted ratios
    pred_ratio_high = predicted_data[10.95]['scission'] / predicted_data[10.95]['crosslink'] # -> [2.0, 2.0]
    pred_ratio_low = predicted_data[0.0108]['scission'] / predicted_data[0.0108]['crosslink'] # -> [2.5, 2.0]
    
    # Manually calculate expected MSE of log ratios
    true_data = env.true_data
    true_ratio_high = true_data[10.95]['scission'] / np.maximum(true_data[10.95]['crosslink'], 1e-9)
    true_ratio_low = true_data[0.0108]['scission'] / np.maximum(true_data[0.0108]['crosslink'], 1e-9)
    
    log_error_high = np.log(pred_ratio_high) - np.log(true_ratio_high)
    log_error_low = np.log(pred_ratio_low) - np.log(true_ratio_low)
    
    total_squared_error = np.sum(log_error_high**2) + np.sum(log_error_low**2)
    num_points = len(log_error_high) + len(log_error_low)
    expected_mse = total_squared_error / num_points
    
    expected_reward = -expected_mse
    
    calculated_reward = env._calculate_reward(predicted_data)
    
    assert np.isclose(calculated_reward, expected_reward)