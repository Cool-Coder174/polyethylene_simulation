"""
Tests for the PolymerSimulationEnv, focusing on the LAMMPS integration.
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
def env(config):
    """Fixture to create a fresh instance of the PolymerSimulationEnv for each test."""
    return PolymerSimulationEnv(config, experimental_data_path="tests/experimental_data.json")

def test_environment_initialization(env, config):
    """
    Tests if the environment initializes correctly with the given configuration.
    """
    assert env.config == config
    assert np.array_equal(env.param_multipliers, [1.0, 1.0]), "Initial multipliers should be [1.0, 1.0]."

def test_reset_method(env):
    """
    Tests if the reset method correctly resets the parameter multipliers and returns the initial state.
    """
    # Modify the state, then reset
    env.param_multipliers = np.array([2.0, 3.0])
    initial_state, info = env.reset()
    
    assert np.array_equal(initial_state, [1.0, 1.0]), "Reset should return the initial state [1.0, 1.0]."
    assert np.array_equal(env.param_multipliers, [1.0, 1.0]), "Multipliers should be reset to [1.0, 1.0]."

def test_step_method_and_action_effect(env):
    """
    Tests that the step method correctly applies an action, updates multipliers,
    and returns a valid state, reward, and done flag.
    """
    env.reset()
    
    # Action should be a small multiplicative change
    action = np.array([0.1, -0.1])
    next_state, reward, done, _, info = env.step(action)
    
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
    # Mock predicted data (since we are not running a real LAMMPS simulation in the test)
    predicted_data = {
        10.95: {'scission': np.array([0.1, 0.2]), 'crosslink': np.array([0.05, 0.1])},
        0.0108: {'scission': np.array([0.05, 0.1]), 'crosslink': np.array([0.02, 0.05])}
    }
    
    # Manually calculate expected MSE of log ratios
    true_data = env.true_data
    true_ratio_high = true_data[10.95]['scission'] / np.maximum(true_data[10.95]['crosslink'], 1e-9)
    true_ratio_low = true_data[0.0108]['scission'] / np.maximum(true_data[0.0108]['crosslink'], 1e-9)
    
    pred_ratio_high = predicted_data[10.95]['scission'] / predicted_data[10.95]['crosslink']
    pred_ratio_low = predicted_data[0.0108]['scission'] / predicted_data[0.0108]['crosslink']
    
    log_error_high = np.log(pred_ratio_high) - np.log(true_ratio_high)
    log_error_low = np.log(pred_ratio_low) - np.log(true_ratio_low)
    
    total_squared_error = np.sum(log_error_high**2) + np.sum(log_error_low**2)
    num_points = len(log_error_high) + len(log_error_low)
    expected_mse = total_squared_error / num_points
    
    expected_reward = -expected_mse
    
    calculated_reward = env._calculate_reward(predicted_data)
    
    assert np.isclose(calculated_reward, expected_reward)
