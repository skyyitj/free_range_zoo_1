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
    """Agent that coordinates with others to suppress wildfires effectively."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)
        self.max_intensity = 10  # Example threshold for maximum fire intensity
        self.resource_threshold = 0.2  # Minimum resource to avoid depletion

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.

        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        has_suppressant = self.observation['self'][:, 3] != 0
        self_x = self.observation['self'][:, 1]
        self_y = self.observation['self'][:, 0]
        self_fire_power = self.observation['self'][:, 2]
        self_suppressant = self.observation['self'][:, 3]

        for batch in range(self.parallel_envs):
            max_score = -100
            max_index = -1

            # No fires, so noop
            if len(self.argmax_store[batch]) == 0:
                self.actions[batch].fill_(-1)
                continue

            # Consider actions for each task in the environment
            for index in range(self.argmax_store[batch].size(0)):
                fire_x = self.argmax_store[batch][index][1]
                fire_y = self.argmax_store[batch][index][0]
                fire_intensity = self.argmax_store[batch][index][2]
                fire_distance = torch.sqrt((fire_x - self_x[batch]) ** 2 + (fire_y - self_y[batch]) ** 2)

                # Define priority for actions: prioritize intensity, proximity, and resource usage
                if fire_intensity >= self.max_intensity or self_suppressant[batch] <= self.resource_threshold:
                    score = 0  # Don't take action if resources are too low or fire is already very intense
                else:
                    # Weighted scoring function
                    score = (2 / ((0.5 * fire_distance + 1) * (0.1 * fire_intensity + 4)))

                if score > max_score:
                    max_score = score
                    max_index = index

            if max_index == -1:
                self.actions[batch].fill_(-1)  # No fires to fight, noop

            self.actions[batch, 0] = max_index
            self.actions[batch, 1] = 0  # Default to 0 as action (no additional task)

        # Agents without suppressant should not take any action
        self.actions[:, 1].masked_fill_(~has_suppressant, -1)

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
        except Exception as e:
            self.observation = observation

        # Extract fire data: position, intensity, fire level
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]  # y, x, intensity
        self.argmax_store = torch.zeros_like(self.fires)

        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
