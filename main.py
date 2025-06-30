# main.py
import numpy as np
import torch
import os
from pathlib import Path
import uuid
import time as timer

from td3_agent import TD3
from replay_buffer import ReplayBuffer
from utils import run_simulation, calculate_reward_and_next_state
from plotting import generate_and_save_plots

# --- Configuration ---
# RL Agent Config
STATE_DIM = 6  # [Lchain, Nchain, time, dose_rate, final_connectivity, final_l2]
ACTION_DIM = 4 # The parameters the agent can control: [Lchain, Nchain, time, dose_rate]
MAX_ACTION = 1.0 # The max value for a normalized action

# Training Loop Config
EPISODES = 500
BATCH_SIZE = 64 # The number of transitions to sample from the replay buffer for each learning step.

# --- Directory Setup ---
# Create directories to store models, simulation data, and plots
Path("models").mkdir(exist_ok=True)
OUTPUT_DIR = Path("simulation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Helper Functions for Parameter Scaling ---
def normalize_state(state_array):
    """Normalizes the state vector to be in a range suitable for the neural network."""
    # [Lchain, Nchain, time, dose_rate, avg_conn, l2]
    bounds = np.array([1000, 500, 365, 2.0, 10.0, 0.1])
    return state_array / bounds

def denormalize_action(action_array):
    """Converts the agent's normalized action back into simulation parameters."""
    # Define the min and max bounds for each parameter
    # [Lchain, Nchain, time, dose_rate]
    low = np.array([10, 5, 30, 0.1])
    high = np.array([500, 100, 365, 1.5])
    
    # Scale the action from [-1, 1] to [0, 1]
    action_array = (action_array + 1) / 2
    
    # Linearly interpolate to the parameter space
    params = low + action_array * (high - low)
    
    # Ensure integer parameters are cast correctly
    params[0] = int(params[0]) # Lchain
    params[1] = int(params[1]) # Nchain
    params[2] = int(params[2]) # time
    
    return {
        "Lchain": params[0],
        "Nchain": params[1],
        "time": params[2],
        "dose_rate": params[3]
    }

# --- Main Execution ---
def main():
    """Main function to run the RL training loop."""
    # Initialize the agent and replay buffer
    agent = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)
    replay_buffer = ReplayBuffer()

    # Try to load a pre-trained model
    try:
        agent.load("models/td3_polymer")
        print("Pre-trained model loaded.")
    except FileNotFoundError:
        print("No pre-trained model found. Starting from scratch.")

    # Initialize the first state. It can be based on a set of default parameters.
    initial_params = {"Lchain": 100, "Nchain": 50, "time": 90, "dose_rate": 0.74}
    # A placeholder state to begin
    state = normalize_state(np.array([100, 50, 90, 0.74, 2.0, 0.01]))

    print("--- Starting Intelligent Simulation Control ---")
    for episode in range(EPISODES):
        start_time = timer.time()
        
        # 1. Agent selects an action based on the current state
        # Add some noise for exploration
        action = agent.select_action(state)
        noise = np.random.normal(0, 0.1, size=ACTION_DIM)
        action = (action + noise).clip(-MAX_ACTION, MAX_ACTION)

        # 2. Map the discrete action to a change in simulation parameters
        sim_params = denormalize_action(action)

        # 3. Run the simulation environment with the new parameters
        print(f"\n[Episode {episode+1}/{EPISODES}] Running simulation with params: {sim_params}")
        
        # A unique ID for this specific simulation run
        simulation_run_id = str(uuid.uuid4())
        
        try:
            # The simulation function runs the model and returns the results DataFrame
            results_df = run_simulation(sim_params, simulation_run_id, OUTPUT_DIR)
            
            # 4. Calculate reward and the next state from the simulation output
            reward_value, next_state_raw = calculate_reward_and_next_state(results_df, sim_params, start_time)
            reward = torch.tensor([reward_value], dtype=torch.float32)
            next_state = normalize_state(next_state_raw)
            done = False

        except Exception as e:
            # Handle simulation crashes or errors with a large penalty
            print(f"--- ERROR in Episode {episode+1}: Simulation failed! {e} ---")
            reward = torch.tensor([-200.0], dtype=torch.float32) # Large penalty
            next_state = None # Terminal state
            done = True

        # 5. Store the transition in the agent's replay buffer
        replay_buffer.add(state, next_state, action, reward.numpy(), float(done))

        # 6. Update the current state
        state = next_state if not done else normalize_state(np.array([100, 50, 90, 0.74, 2.0, 0.01])) # Reset on failure

        # 7. Perform one step of learning on the agent
        if len(replay_buffer.storage) > BATCH_SIZE:
            agent.train(replay_buffer, BATCH_SIZE)

        # 8. Generate and save correlation plots from the latest simulation data
        if not done:
            print("Generating correlation plots...")
            generate_and_save_plots(results_df, OUTPUT_DIR)

        # 9. Save the agent's model periodically
        if (episode + 1) % 25 == 0:
            print(f"\n--- Saving model at episode {episode+1} ---")
            agent.save("models/td3_polymer")

        print(f"Episode {episode+1} finished. Reward: {reward.item():.2f}. Time: {timer.time() - start_time:.2f}s")
        
    print("\n--- Training complete. ---")

if __name__ == "__main__":
    main()

