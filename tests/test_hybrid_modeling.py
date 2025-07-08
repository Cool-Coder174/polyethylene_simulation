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

@pytest.fixture(scope="module")
def dummy_scission_model(tmpdir_factory):
    """Fixture to create a dummy scission model file for testing."""
    models_dir = Path(tmpdir_factory.mktemp("models"))
    equation_file = models_dir / "scission_equation.json"
    
    # A simple equation: 0.1 * x0 + 0.01 * x1 (dose_rate, time)
    equation_details = {
        "latex": "0.1 x_{0} + 0.01 x_{1}",
        "sympy_expr": "0.1*x0 + 0.01*x1",
        "lambda_format": "lambda x0, x1: (0.1*x0 + 0.01*x1)",
        "tree": {
            "nodes": [
                {"feature": 0, "op": "+", "left": 1, "right": 2},
                {"feature": 0, "op": "*", "left": 3, "right": 4},
                {"feature": 1, "op": "*", "left": 5, "right": 6},
                {"feature": 0, "op": "const", "value": 0.1},
                {"feature": 0, "op": "variable", "value": 0},
                {"feature": 0, "op": "const", "value": 0.01},
                {"feature": 1, "op": "variable", "value": 1}
            ]
        }
    }
    with open(equation_file, 'w') as f:
        json.dump(equation_details, f)
    
    return equation_file

def test_scission_equation_loading(config, dummy_scission_model):
    """
    Tests that the PolymerSimulationEnv can correctly load and evaluate
    the symbolic scission equation from a JSON file.
    """
    # Temporarily point to the dummy model file
    config['output_paths']['model_path'] = str(Path(dummy_scission_model).parent)
    
    # To load the dummy model, we need to trick the env into looking at the temp dir
    # A bit of a hack: we can't easily change the hardcoded path in the env,
    # so we will rely on the fact that the test runs from the root project dir.
    # We will create a dummy models/scission_equation.json file.
    
    Path("models").mkdir(exist_ok=True)
    with open("models/scission_equation.json", 'w') as f:
        with open(dummy_scission_model, 'r') as dummy_f:
            f.write(dummy_f.read())

    env = PolymerSimulationEnv(config)
    
    assert env.scission_function is not None
    
    # Test evaluation
    dose_rate = 10.0
    time = 100.0
    # Expected: 0.1 * 10.0 + 0.01 * 100.0 = 1.0 + 1.0 = 2.0
    expected_rate = 2.0
    
    # The symbolic regression uses x0, x1. Let's manually call it.
    rate = env.evaluate_scission_equation(dose_rate, time)
    
    assert np.isclose(rate, expected_rate, atol=1e-5)

def test_env_step_and_reward(config):
    """
    Tests that the environment can take a step and return a finite reward.
    """
    env = PolymerSimulationEnv(config)
    
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

