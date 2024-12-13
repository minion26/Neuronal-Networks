import random
from collections import deque

# This is where the experience replay memory is stored
# Experience replay is a method used to store experiences of an agent in a memory buffer, a deque
# The agent can then sample from this memory buffer to train itself



class ReplayMemory() :
    def __init__(self, maxLen, seed=None):
        self.memory = deque(maxlen=maxLen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition) # transition is a tuple of (state, action, next_state, reward, terminated)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
