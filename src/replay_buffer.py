"""
This module implements a Replay Buffer, a fundamental component in many
reinforcement learning algorithms, particularly off-policy methods like SAC.
It stores experiences (state, action, reward, next_state, done) collected
by the agent, allowing for efficient sampling of past transitions for training.
"""

import numpy as np
import random

class ReplayBuffer:
    """
    A simple replay buffer to store and sample experiences for reinforcement learning.

    The buffer stores tuples of (state, next_state, action, reward, done) and
    allows for random sampling of batches, which helps to break correlations
    between consecutive samples and improve learning stability.
    """
    def __init__(self, max_size: int = 1_000_000):
        """
        Initializes the ReplayBuffer.

        Args:
            max_size (int): The maximum number of experiences the buffer can store.
                            When the buffer is full, new experiences will overwrite
                            the oldest ones in a circular manner.
        """
        self.storage = [] # List to store the experiences
        self.max_size = max_size # Maximum capacity of the buffer
        self.ptr = 0 # Pointer to the current position for adding new experiences

    def add(self, state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: float, done: bool):
        """
        Adds a new experience to the replay buffer.

        If the buffer is not yet full, the experience is appended. Once the buffer
        reaches its `max_size`, new experiences will overwrite the oldest ones
        in a circular fashion (FIFO - First-In, First-Out).

        Args:
            state (np.ndarray): The observation of the environment before the action.
            next_state (np.ndarray): The observation of the environment after the action.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received after taking the action.
            done (bool): A boolean indicating whether the episode terminated after this action.
        """
        data = (state, next_state, action, reward, done)
        if len(self.storage) < self.max_size:
            # If buffer is not full, append the new experience.
            self.storage.append(data)
        else:
            # If buffer is full, overwrite the oldest experience.
            self.storage[self.ptr] = data
            # Move the pointer to the next position, wrapping around if necessary.
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing five NumPy arrays:
                   - states (np.ndarray): A batch of states.
                   - next_states (np.ndarray): A batch of next states.
                   - actions (np.ndarray): A batch of actions.
                   - rewards (np.ndarray): A batch of rewards.
                   - dones (np.ndarray): A batch of done flags.
        """
        # Randomly select `batch_size` experiences from the stored data.
        batch = random.sample(self.storage, batch_size)
        
        # Unzip the batch of tuples into separate lists for each component
        # and then stack them into NumPy arrays.
        state, next_state, action, reward, done = map(np.stack, zip(*batch))
        
        return state, next_state, action, reward, done