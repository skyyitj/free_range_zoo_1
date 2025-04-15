# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class FirefightingAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)
        self.max_distance = 5  # Max distance to engage with a fire
        self.suppressant_threshold = 0.2  # Threshold to start using suppressant

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        self_x = self.observation['self'][:, 1]
        self_y = self.observation['self'][:, 0]
        self_suppressant = self.observation['self'][:, 3]
        self_capacity = self.observation['self'][:, 2]
        self_fires = self.observation['tasks'][:, :, [0, 1, 3, 4]]  # Fire position, level, and intensity

        # Iterate over all parallel environments
        for batch in range(self.parallel_envs):
            max_score = -float('inf')
            best_fire_idx = -1
            action = -1

            # Look for the fire with the highest intensity within the max distance
            for idx, fire in enumerate(self_fires[batch]):
                fire_x, fire_y, fire_level, fire_intensity = fire
                # Calculate distance to the fire
                distance = torch.sqrt((fire_x - self_x[batch]) ** 2 + (fire_y - self_y[batch]) ** 2)

                # Skip fires that are too far away (based on max_distance)
                if distance > self.max_distance:
                    continue

                # If the fire is within range, calculate a score based on intensity and distance
                distance_score = 1 / (distance + 1)
                intensity_score = fire_intensity / (fire_level + 1)
                score = distance_score * intensity_score  # Prioritize intense and nearby fires

                # Determine if this fire is the best option based on current score
                if score > max_score:
                    max_score = score
                    best_fire_idx = idx
                    action = 0  # Default action to fight the fire

            # If no valid fire found, agent will do nothing (noop)
            if best_fire_idx == -1:
                self.actions[batch].fill_(-1)
            else:
                # If the agent has sufficient suppressant and capacity, take action
                if self_suppressant[batch] > self.suppressant_threshold and self_capacity[batch] > 0:
                    self.actions[batch, 0] = best_fire_idx
                    self.actions[batch, 1] = action  # Fight fire with suppressant
                else:
                    self.actions[batch].fill_(-1)  # No suppressant or capacity left, noop

        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation = observation
        self.fires = self.observation['tasks'][:, :, [0, 1, 3, 4]]  # Fire position, fire level, fire intensity
        self.argmax_store = torch.zeros_like(self.fires)

        # Store fire details for future actions
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
