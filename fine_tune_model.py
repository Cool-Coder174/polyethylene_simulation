
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
import logging
import uuid
from datetime import datetime

from src.polymer_env import PolymerSimulationEnv
from src.sac_agent import SAC
from src.replay_buffer import ReplayBuffer
import src.database as db
from src.interactive_plotting import create_interactive_plots

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fine_tune_model():
    """
    Main function to run the fine-tuning process.
    """
    # Load configuration
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    db_path = Path(config['output_paths']['database_path'])
    db.init_database(db_path)
    logging.info("--- Phase 1: Discovering Scission Model via Symbolic Regression ---")
    try:
        subprocess.run([sys.executable, "-c", "import pysr; pysr.install()"], check=True)
        subprocess.run([sys.executable, "scripts/discover_scission_model.py"], check=True)
        logging.info("Symbolic regression complete. Scission model saved.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error during symbolic regression: {e}")
        logging.error("Please ensure Julia is installed and accessible.")
        logging.info("Attempting to proceed without re-running symbolic regression...")
        if not Path("models/scission_equation.json").exists():
            logging.error("Error: models/scission_equation.json not found. Cannot proceed.")
            return

    logging.info("\n--- Phase 2: Fine-Tuning Parameters with Reinforcement Learning ---")
    
    # Initialize environment, agent, and replay buffer
    env = PolymerSimulationEnv(config)
    agent = SAC(config)
    replay_buffer = ReplayBuffer()

    state, _ = env.reset()
    total_steps = config['rl_hyperparameters']['episodes'] # Use episodes from config
    
    run_id = str(uuid.uuid4())
    start_time = datetime.now()
    initial_dose_rate = config['run_parameters']['param_bounds']['dose_rate'][0]

    logging.info(f"Starting DSAC training for {total_steps} steps...")
    for step in range(total_steps):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        replay_buffer.add(state, next_state, action, reward, float(done))
        state = next_state

        if len(replay_buffer.storage) > config['rl_hyperparameters']['batch_size']:
            agent.train(replay_buffer, config['rl_hyperparameters']['batch_size'])

        if (step + 1) % config['rl_hyperparameters']['checkpoint_frequency'] == 0:
            logging.info(f"Step: {step + 1}/{total_steps}, Reward: {reward:.4f}")
            logging.info(f"  Current Multipliers: {env.param_multipliers}")
            
            # Log data to database
            log_df = env.get_simulation_data_df()
            if not log_df.empty:
                # Add run_id, episode_num to the DataFrame before logging
                log_df['run_id'] = run_id
                log_df['episode_num'] = step + 1
                # Add dummy values for other columns expected by simulation_data table
                log_df['voxel_id'] = 0 # Assuming single voxel for now
                log_df['pe_conc'] = 0.0
                log_df['o2_conc'] = 0.0
                log_df['pe_rad_conc'] = 0.0
                log_df['peoo_rad_conc'] = 0.0
                log_df['peooh_conc'] = 0.0
                db.log_simulation_data(db_path, log_df)

    logging.info("Training complete.")

    # --- Phase 3: Save Results and Validate ---
    logging.info("\n--- Phase 3: Saving Final Parameters and Validation ---")
    
    # Save the final tuned parameter multipliers
    final_multipliers = {"crosslink_multiplier": float(env.param_multipliers[0]),
                         "scission_multiplier": float(env.param_multipliers[1])}
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "param_multipliers.json", 'w') as f:
        json.dump(final_multipliers, f, indent=4)
    logging.info(f"Final multipliers saved to {models_dir / 'param_multipliers.json'}")

    # Log final metadata
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate final reward (assuming the last reward is representative or average over last few steps)
    # For a single-step episode, the last reward is the final reward.
    final_reward = reward 

    metadata = {
        "run_id": run_id,
        "optuna_trial_id": None, # Not an Optuna run
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "initial_dose_rate": initial_dose_rate,
        "final_reward": final_reward,
        "status": "completed"
    }
    db.log_run_metadata(db_path, metadata)

    # Create validation plots
    plot_dir = Path(config['output_paths']['plot_path'])
    plot_dir.mkdir(parents=True, exist_ok=True)
    create_interactive_plots(db_path, plot_dir, run_id)
    logging.info(f"Validation plots saved to {plot_dir}")

if __name__ == "__main__":
    fine_tune_model()
