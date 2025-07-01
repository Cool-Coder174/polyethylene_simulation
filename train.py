# train.py
import yaml
import uuid
import time
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from polymer_env import PolymerSimulationEnv
from td3_agent import TD3
from replay_buffer import ReplayBuffer
from database import init_database, log_simulation_data, log_run_metadata
from interactive_plotting import create_interactive_plots

# --- Load Configuration ---
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

db_path = Path(config['paths']['database_file'])
model_dir = Path(config['paths']['model_dir'])
plot_dir = Path(config['paths']['plot_dir'])
model_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)

# --- Objective Function for Optuna ---
def objective(trial):
    """The function Optuna will try to optimize."""
    # Suggest hyperparameters for the TD3 agent
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    
    # Initialize environment and agent with suggested params
    env = PolymerSimulationEnv(config)
    agent = TD3(
        state_dim=config['agent']['state_dim'],
        action_dim=config['agent']['action_dim'],
        max_action=config['agent']['max_action'],
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau
    )
    replay_buffer = ReplayBuffer()
    
    total_reward = 0
    state = env.reset()

    # --- Main Training Loop for this Trial ---
    for episode in range(config['training']['episodes']):
        run_id = str(uuid.uuid4())
        start_time = datetime.now()

        action = agent.select_action(np.array(state))
        noise = np.random.normal(0, 0.1, size=config['agent']['action_dim'])
        action = (action + noise).clip(-config['agent']['max_action'], config['agent']['max_action'])

        next_state, reward, done, results_df = env.step(action)
        
        # Log results to database
        results_df['run_id'] = run_id
        results_df['optuna_trial_id'] = trial.number
        results_df['episode_num'] = episode
        log_simulation_data(db_path, results_df)
        
        log_run_metadata(db_path, {
            "run_id": run_id, "optuna_trial_id": trial.number, "episode_num": episode,
            "start_time": start_time.isoformat(), "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "initial_params": json.dumps(env.params), "final_reward": reward
        })
        
        replay_buffer.add(state, next_state, action, reward, float(done))
        state = next_state

        if len(replay_buffer.storage) > config['training']['batch_size']:
            agent.train(replay_buffer, config['training']['batch_size'])
        
        total_reward += reward

        # Pruning for bad Optuna trials
        trial.report(total_reward, episode)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Save the final trained model for this trial
    agent.save(model_dir / f"td3_trial_{trial.number}")
    return total_reward

# --- Main Execution ---
if __name__ == "__main__":
    init_database(db_path)
    
    if config['training']['enable_optuna']:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=config['training']['optuna_trials'])
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        # Run a single training session without Optuna
        class MockTrial:
            def suggest_float(self, name, low, high, log=False): return {"lr": 1e-3, "gamma": 0.99, "tau": 0.005}[name]
            def report(self, val, step): pass
            def should_prune(self): return False
            @property
            def number(self): return 0
        
        objective(MockTrial())

