# main.py
import numpy as np
import torch
import os
import pandas as pd
from td3_agent import TD3
from replay_buffer import ReplayBuffer
from utils import simulate_graph, save_csv_data, make_graphs

# Setup
state_dim = 5  # [time, Lchain, Nchain, trials, dose rate]
action_dim = 5  # same
max_action = 1

agent = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

# Training loop config
EPISODES = 500
BATCH_SIZE = 64
RUNS_PER_EPISODE = 5

def normalize_input(state):
    norm = [300, 1000, 10, 10, 1]
    return np.array(state) / np.array(norm)

def denormalize_action(action):
    bounds = [300, 1000, 10, 10, 1]  # max values
    return np.clip(np.array(action) * np.array(bounds), [1, 1, 1, 1, 0.01], bounds)

for episode in range(EPISODES):
    state = np.random.uniform(low=0.1, high=1.0, size=state_dim)
    for run in range(RUNS_PER_EPISODE):
        action = agent.select_action(state)
        params = denormalize_action(action)
        
        result, reward, next_state = simulate_graph(params)

        done = 0.0 if run < RUNS_PER_EPISODE - 1 else 1.0
        replay_buffer.add(state, next_state, action, reward, done)
        state = next_state

        # Save per-run CSV
        save_csv_data(result)
        make_graphs(result)

    if len(replay_buffer.storage) >= BATCH_SIZE:
        agent.train(replay_buffer, BATCH_SIZE)

    if episode % 50 == 0:
        agent.save(f"models/td3_ep{episode}")

print("Training complete.")

