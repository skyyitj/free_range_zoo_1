# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class FirefighterAgent(Agent):
    """Agent that coordinates firefighting efforts in a grid environment."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        has_suppressant = self.observation['self'][:, 3] != 0  # Check if the agent has suppressant
        self_x = self.observation['self'][:, 1]  # Agent's x position
        self_y = self.observation['self'][:, 0]  # Agent's y position
        self_capacity = self.observation['self'][:, 2]  # Agent's fire power capacity
        self_suppressant = self.observation['self'][:, 3]  # Agent's suppressant capacity

        for batch in range(self.parallel_envs):
            maximum_score = -float('inf')
            maximum_index = -1
            best_action = -1  # Default action is 'noop'

            # If no fire to suppress, do nothing
            if len(self.argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)  # Noop
                continue

            # Iterate over all fires to find the highest priority target (based on fire intensity and distance)
            for index in range(self.argmax_store[batch].size(0)):
                fire_x = self.argmax_store[batch][index][1]
                fire_y = self.argmax_store[batch][index][0]
                fire_intensity = self.argmax_store[batch][index][2]
                fire_distance = torch.sqrt((self_x[batch] - fire_x)**2 + (self_y[batch] - fire_y)**2)

                # Score calculation based on intensity, distance, and available resources
                intensity_weight = 0.5
                distance_weight = 0.3
                resource_weight = 0.2

                # Penalize if fire power or suppressant is low
                if self_capacity[batch] < 0.1 or self_suppressant[batch] < 0.1:
                    resource_score = 0  # Can't fight fires if out of resources
                else:
                    resource_score = 1  # Sufficient resources

                # Compute the score for this fire and action
                score = (intensity_weight * fire_intensity - distance_weight * fire_distance + resource_weight * resource_score)

                # Update the best fire to fight if this one has a higher score
                if score > maximum_score:
                    maximum_score = score
                    maximum_index = index

            if maximum_index == -1:
                self.actions[batch].fill_(-1)  # No action to take
            else:
                self.actions[batch, 0] = maximum_index  # Assign the action to fight the selected fire
                self.actions[batch, 1] = 0  # Default action is fighting the fire (no movement)

            # Ensure that agents without suppressant can't perform actions that require it
            self.actions[batch, 1].masked_fill_(~has_suppressant[batch], -1)

        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        try:
            self.observation, self.t_mapping = observation
            self.t_mapping = self.t_mapping['agent_action_mapping']
        except Exception as e:  # Catch all exceptions to avoid crashing
            self.observation = observation

        # Extract fire-related information from the observations
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        self.argmax_store = torch.zeros_like(self.fires)

        # Store fire information (y, x, intensity) for each parallel environment
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
