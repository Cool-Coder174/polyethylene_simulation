"""
Tests for the hybrid symbolic-RL modeling components.
"""
import pytest
import numpy as np
import json
from pathlib import Path
import yaml

from src.polymer_env import PolymerSimulationEnv

@pytest.fixture(scope="module")
def config():
    """Fixture to load the main configuration."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def dummy_scission_model_path(tmp_path):
    """Fixture to create a dummy scission model file in a temporary directory."""
    equation_file = tmp_path / "scission_equation.json"
    
    # A simple equation: 0.1 * x0 + 0.01 * x1 (dose_rate, time)
    equation_details = {
        "latex": "0.1 x_{0} + 0.01 x_{1}",
        "sympy_expr": "0.1*x0 + 0.01*x1",
    }
    with open(equation_file, 'w') as f:
        json.dump(equation_details, f)
    
    return str(equation_file)

def test_scission_equation_loading(config, dummy_scission_model_path):
    """
    Tests that the PolymerSimulationEnv can correctly load and evaluate
    the symbolic scission equation from a JSON file.
    """
    env = PolymerSimulationEnv(config, scission_model_path=dummy_scission_model_path)
    
    assert env.scission_function is not None
    
    # Test evaluation
    dose_rate = 10.0
    time = 100.0
    # Expected: 0.1 * 10.0 + 0.01 * 100.0 = 1.0 + 1.0 = 2.0
    expected_rate = 2.0
    
    rate = env.evaluate_scission_equation(dose_rate, time)
    
    assert np.isclose(rate, expected_rate, atol=1e-5)

def test_env_step_and_reward(config, dummy_scission_model_path):
    """
    Tests that the environment can take a step and return a finite reward.
    """
    env = PolymerSimulationEnv(config, scission_model_path=dummy_scission_model_path)
    
    # Use the action space defined in the env for a valid random action
    action_low = env.action_space[0]
    action_high = env.action_space[1]
    random_action = np.random.uniform(low=action_low, high=action_high, size=(2,))
    
    obs, reward, done, info = env.step(random_action)
    
    assert np.isfinite(reward)
    assert isinstance(reward, float)
    
    assert obs.shape == (2,)
    assert np.all(obs >= env.observation_space[0])
    assert np.all(obs <= env.observation_space[1])
    
    assert done is True

