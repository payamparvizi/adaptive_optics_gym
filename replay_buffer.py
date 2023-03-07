"""
        This file presents the storage of the trajectories of experiences
"""


#----------------------------- Importing modules -----------------------------#
import random
import numpy as np


#------------------------------ Initialization -------------------------------#
class ReplayBuffer:
    def __init__(self, capacity: int, a_dim: int, a_dtype, s_dim: int, s_dtype, store_mu: bool=False) -> None:
        self.capacity = capacity
        self.buffer = []
        self.position = 0


#--------------------------------- Addition ----------------------------------#
# Storing the trajectories of experiences
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity


#--------------------------------- Sampling ----------------------------------#
# randomly sample a batch of transitions from replay buffer
    def sample_batch(self, batch_size, seed_):
        random.seed(seed_)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


#---------------------------------- Legth ------------------------------------#
# calculate the length of the replay buffer
    def __len__(self):
        return len(self.buffer)


#-----------------------------------------------------------------------------#