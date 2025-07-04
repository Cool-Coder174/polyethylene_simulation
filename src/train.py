"""
This module orchestrates the training of the reinforcement learning agent
for the polyethylene degradation simulation. It integrates the simulation
environment, the SAC agent, replay buffer, and database logging.

It also supports hyperparameter optimization using Optuna to find the best
agent configurations.
"""

import yaml
import uuid
import time
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import torch

# Import components from the src directory
from src.polymer_env import PolymerSimulationEnv
from src.sac_agent import SAC
from src.replay_buffer import ReplayBuffer
from src.database import init_database, log_simulation_data, log_run_metadata
from src.interactive_plotting import create_interactive_plots

# --- Load Configuration ---
# The main configuration file (config.yaml) contains all parameters
# for the simulation, RL agent, and training process.
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Define and create output directories for database, models, and plots.
db_path = Path(config['output_paths']['database_path'])
model_dir = Path(config['output_paths']['model_path'])
plot_dir = Path(config['output_paths']['plot_path'])
model_dir.mkdir(parents=True, exist_ok=True) # Create model directory if it doesn't exist
plot_dir.mkdir(parents=True, exist_ok=True)   # Create plot directory if it doesn't exist

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    The objective function that Optuna will try to optimize.
    Each call to this function represents a single trial (a full training run)
    with a specific set of hyperparameters suggested by Optuna.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest
                                    hyperparameters and report intermediate results.

    Returns:
        float: The final reward achieved by the agent in this trial, which Optuna
               aims to maximize.
    """
    run_id = str(uuid.uuid4()) # Generate a unique ID for this simulation run
    start_time = datetime.now() # Record the start time of the trial
    status = "running" # Initial status of the run
    initial_dose_rate = None # Initialize, will be set after env reset
    final_reward = 0.0 # Initialize final reward

    # Log run metadata at the start of the trial. This helps track ongoing runs.
    log_run_metadata(db_path, {
        "run_id": run_id,
        "optuna_trial_id": trial.number, # Optuna's unique trial number
        "start_time": start_time.isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "initial_dose_rate": initial_dose_rate,
        "final_reward": final_reward,
        "status": status
    })

    try:
        # --- Hyperparameter Suggestion by Optuna ---
        # Optuna suggests values for key hyperparameters of the SAC agent.
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        tau = trial.suggest_float("tau", 0.001, 0.02)
        alpha = trial.suggest_float("alpha", 0.1, 0.3)

        # Update the config with the suggested hyperparameters for this trial.
        # This ensures the SAC agent is initialized with the trial's specific parameters.
        trial_config = config.copy()
        trial_config['rl_hyperparameters']['learning_rate'] = learning_rate
        trial_config['rl_hyperparameters']['gamma'] = gamma
        trial_config['rl_hyperparameters']['tau'] = tau
        trial_config['rl_hyperparameters']['alpha'] = alpha

        # Initialize environment, agent, and replay buffer for the current trial.
        env = PolymerSimulationEnv(trial_config)
        agent = SAC(trial_config) # Pass the updated config to the SAC agent
        replay_buffer = ReplayBuffer()

        total_reward = 0.0 # Accumulator for the total reward in this trial
        state = env.reset() # Reset the environment to get the initial state
        initial_dose_rate = env.dose_rate # Capture the initial dose rate after environment reset

        # Update the run_metadata with the actual initial dose rate.
        log_run_metadata(db_path, {
            "run_id": run_id,
            "optuna_trial_id": trial.number,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "initial_dose_rate": initial_dose_rate,
            "final_reward": None,
            "status": status
        })

        # --- Main Training Loop for this Trial ---
        # The agent interacts with the environment for a specified number of episodes.
        for episode in range(config['rl_hyperparameters']['episodes']):
            # Agent selects an action based on the current state.
            action = agent.select_action(np.array(state))

            # Environment takes a step with the selected action.
            # The `env.step` returns next_state, reward, done, and an info dictionary.
            # The info dictionary is currently empty and does not contain `results_df`.
            next_state, reward, done, _ = env.step(action)

            # Add the experience (state, next_state, action, reward, done) to the replay buffer.
            replay_buffer.add(state, next_state, action, reward, float(done))
            state = next_state # Update the current state for the next iteration

            # If enough experiences are in the replay buffer, train the agent.
            if len(replay_buffer.storage) > config['rl_hyperparameters']['batch_size']:
                agent.train(replay_buffer, config['rl_hyperparameters']['batch_size'])

            total_reward += reward # Accumulate total reward for the trial

            # --- Checkpointing ---
            # Save the agent's models periodically.
            if episode % config['rl_hyperparameters']['checkpoint_frequency'] == 0:
                agent.save(model_dir / f"sac_trial_{trial.number}_episode_{episode}")

            # --- Optuna Pruning ---
            # Report intermediate reward to Optuna. Optuna uses this to decide
            # whether to prune (stop early) unpromising trials.
            trial.report(total_reward, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned() # Raise exception to signal Optuna to prune

        status = "completed" # Mark trial as completed if it finishes all episodes
        final_reward = total_reward # Set the final reward for logging

    except optuna.exceptions.TrialPruned:
        status = "pruned" # Mark trial as pruned
        final_reward = total_reward # Log the reward up to the point of pruning
        raise # Re-raise the exception to let Optuna handle the pruning
    except Exception as e:
        status = "failed" # Mark trial as failed if an unexpected error occurs
        final_reward = total_reward # Log the reward up to the point of failure
        print(f"Run {run_id} failed with error: {e}")
        raise # Re-raise the exception after logging for debugging
    finally:
        # --- Final Logging of Run Metadata ---
        # This block ensures that run metadata is always logged, regardless of trial outcome.
        end_time = datetime.now() # Record the end time of the trial
        duration = (end_time - start_time).total_seconds() # Calculate total duration
        log_run_metadata(db_path, {
            "run_id": run_id,
            "optuna_trial_id": trial.number,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "initial_dose_rate": initial_dose_rate,
            "final_reward": final_reward,
            "status": status
        })

    # Save the final trained model for this trial if it completed successfully.
    agent.save(model_dir / f"sac_trial_{trial.number}")
    return total_reward # Return the final reward for Optuna optimization

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CUDA System Check ---
    # Check for CUDA (GPU) availability and print relevant information.
    if torch.cuda.is_available():
        print(f"CUDA is available! Using {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (Compute Capability: {torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}) ")
    else:
        print("CUDA is not available. Using CPU.")

    # Initialize the SQLite database (creates tables if they don't exist).
    init_database(db_path)

    # --- Optuna Study or Single Run ---
    if config['rl_hyperparameters']['enable_optuna']:
        # Create an Optuna study to find optimal hyperparameters.
        # direction="maximize" means Optuna will try to maximize the objective function's return value.
        # pruner=optuna.pruners.MedianPruner() helps stop unpromising trials early.
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        # Optimize the objective function for a specified number of trials.
        study.optimize(objective, n_trials=config['rl_hyperparameters']['optuna_trials'])

        # Print the results of the best trial found by Optuna.
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        # --- Single Training Session (without Optuna) ---
        # If Optuna is disabled, run a single training session using default/configured hyperparameters.
        class MockTrial:
            """
            A mock class to simulate an Optuna trial when Optuna is not enabled.
            This allows the `objective` function to be called directly without changes.
            """
            def suggest_float(self, name, low, high, log=False):
                # Return the fixed hyperparameters from the config.
                params = {
                    "lr": config['rl_hyperparameters']['learning_rate'],
                    "gamma": config['rl_hyperparameters']['gamma']['default'], # Use default if specified
                    "tau": config['rl_hyperparameters']['tau']['default'],     # Use default if specified
                    "alpha": config['rl_hyperparameters']['alpha']['default'] # Use default if specified
                }
                return params[name]
            
            def report(self, val, step): 
                pass # No reporting needed for a single run
            
            def should_prune(self): 
                return False # No pruning for a single run
            
            @property
            def number(self):
                return 0 # Always trial 0 for a single run

        # Call the objective function with the mock trial.
        objective(MockTrial())
