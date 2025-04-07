# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class ImprovedAgent(BaseAgent):
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001
        self.epsilon = 0.1  # Exploration rate
        self.policy = {}  # Store the agent's policy
        self.value_function = {}  # State value function for V(s)

    # Optimized policy function (updated)
    def policy_function(self, state):
        """ A simple epsilon-greedy policy with a dynamic adjustment of epsilon. """
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            return self.action_space.sample()
        else:
            # Exploitation: Choose the best action based on the value function
            action_values = self.value_function.get(state, {})
            if not action_values:
                # If no value exists, sample randomly
                return self.action_space.sample()
            return max(action_values, key=action_values.get)

    def act(self, state):
        """ Get the action based on the policy function. """
        action = self.policy_function(state)
        return action

    def observe(self, state, action, reward, next_state, done):
        """ Update the value function based on observed rewards. """
        current_value = self.value_function.get(state, 0)
        next_value = self.value_function.get(next_state, 0)
        
        # Using a simple TD(0) update rule: V(s) <- V(s) + alpha * (reward + gamma * V(s') - V(s))
        self.value_function[state] = current_value + self.learning_rate * (reward + self.gamma * next_value - current_value)

        # Optionally adjust epsilon for exploration-exploitation balance
        if done:
            self.epsilon = max(0.01, self.epsilon * 0.99)  # Decay epsilon over time

# Define the agent with the environment's action and observation space
agent = ImprovedAgent(action_space=env.action_space, observation_space=env.observation_space)
