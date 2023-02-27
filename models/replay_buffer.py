# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:08:18 2022

@author: payam
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:11:51 2022

@author: payam
"""
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, a_dim: int, a_dtype, s_dim: int, s_dtype, store_mu: bool=False) -> None:
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample_batch(self, batch_size, seed_):
        random.seed(seed_)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)