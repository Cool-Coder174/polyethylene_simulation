import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, next_state, action, reward, done):
        data = (state, next_state, action, reward, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, next_state, action, reward, done = map(np.stack, zip(*batch))
        return state, next_state, action, reward, done

