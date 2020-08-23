import numpy as np

class rl_memory(object):

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.states = np.zeros((self.capacity,4))
        self.actions = np.zeros(self.capacity, dtype=np.int)
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity)
        self.next_states = np.zeros((self.capacity, 4))
        self.current = 0

    def add(self, state, action, reward, next_state, done):

        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_state
        self.dones[self.current] = done
        self.current = (self.current + 1) % self.capacity

    def get_batch(self, batch_size):
        indexes = np.random.choice(min(self.capacity, self.current), batch_size, replace=True)
        return self.states[indexes], self.actions[indexes], self.rewards[indexes], self.next_states[indexes],\
               self.dones[indexes]
