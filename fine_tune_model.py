
"""
This script orchestrates the entire hybrid modeling workflow:
1.  It runs the symbolic regression script to discover the scission model.
2.  It fine-tunes the kinetic model's parameters using a Distributional Soft
    Actor-Critic (DSAC) agent.
3.  It saves the final tuned parameters and validation plots.
"""

import yaml
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

from src.polymer_env import PolymerSimulationEnv
from src.sac_agent import SAC
from src.replay_buffer import ReplayBuffer
import src.database as db
# Note: We will need a plotting function, let's assume one exists in interactive_plotting
from src.interactive_plotting import create_interactive_plots

def fine_tune_model():
    """
    Main function to run the fine-tuning process.
    """
    db_path = Path("simulation_results.db")
    db.init_database(db_path)
    print("--- Phase 1: Discovering Scission Model via Symbolic Regression ---")
    # We need to ensure PySR has a Julia environment.
    # This command will instantiate a Julia project environment for PySR.
    # It's better to run this once manually, but we include it here for automation.
    try:
        subprocess.run([sys.executable, "-c", "import pysr; pysr.install()"], check=True)
        subprocess.run([sys.executable, "scripts/discover_scission_model.py"], check=True)
        print("Symbolic regression complete. Scission model saved.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during symbolic regression: {e}")
        print("Please ensure Julia is installed and accessible.")
        print("Attempting to proceed without re-running symbolic regression...")
        if not Path("models/scission_equation.json").exists():
            print("Error: models/scission_equation.json not found. Cannot proceed.")
            return

    print("\n--- Phase 2: Fine-Tuning Parameters with Reinforcement Learning ---")
    
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize environment, agent, and replay buffer
    env = PolymerSimulationEnv(config)
    agent = SAC(config)
    replay_buffer = ReplayBuffer()

    state = env.reset()
    total_steps = 200000
    
    print(f"Starting DSAC training for {total_steps} steps...")
    for step in range(total_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        replay_buffer.add(state, next_state, action, reward, float(done))
        state = next_state

        if len(replay_buffer.storage) > config['rl_hyperparameters']['batch_size']:
            agent.train(replay_buffer, config['rl_hyperparameters']['batch_size'])

        if (step + 1) % 1000 == 0:
            print(f"Step: {step + 1}/{total_steps}, Reward: {reward:.4f}")
            print(f"  Current Multipliers: {env.param_multipliers}")
            # Log data to database
            log_df = env.get_simulation_data_df()
            db.log_simulation_data(db_path, log_df)

    print("Training complete.")

    # --- Phase 3: Save Results and Validate ---
    print("\n--- Phase 3: Saving Final Parameters and Validation ---")
    
    # Save the final tuned parameter multipliers
    final_multipliers = {"crosslink_multiplier": env.param_multipliers[0],
                         "scission_multiplier": env.param_multipliers[1]}
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "param_multipliers.json", 'w') as f:
        json.dump(final_multipliers, f, indent=4)
    print(f"Final multipliers saved to {models_dir / 'param_multipliers.json'}")

    # Log final metadata
    import uuid
    from datetime import datetime
    start_time = datetime.now()
    
    # Placeholder for reward calculation (ensure this is defined earlier in the code)
    reward = env.get_final_reward()  # Replace with actual method to get final reward
    
    end_time = datetime.now()
    metadata = {
        "run_id": str(uuid.uuid4()),  # Generate a unique ID
        "optuna_trial_id": None,
        "episode_num": total_steps,  # Assuming total_steps represents the number of episodes
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "initial_params": config.get("initial_params", {}),  # Replace with actual initial params
        "final_reward": reward
    }
    db.log_run_metadata(db_path, metadata)

    # Run a final validation simulation with the tuned parameters
    # This part is illustrative. A dedicated validation function would be better.
    # For now, we can re-use the logic inside the env's step function.
    
    # Create validation plots
    # This assumes a function `create_comparison_plots` exists and works.
    # We will need to create it.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # I will create a placeholder for the plotting function call
    print("Validation plotting is not yet implemented.")
    # create_interactive_plots(env.true_data, predicted_data, results_dir)

if __name__ == "__main__":
    fine_tune_model()
