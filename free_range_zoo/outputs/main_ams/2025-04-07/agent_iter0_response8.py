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
    """Agent that coordinates firefighting actions to suppress wildfires in a grid environment."""

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
        # Extract agent state (position, resources, etc.)
        has_suppressant = self.observation['self'][:, 3] != 0  # Check if the agent has suppressant
        self_x = self.observation['self'][:, 1]  # Agent's x-position
        self_y = self.observation['self'][:, 0]  # Agent's y-position
        self_fire_power = self.observation['self'][:, 2]  # Agent's firepower
        self_suppressant = self.observation['self'][:, 3]  # Agent's suppressant

        # Initialize actions
        self.actions.fill_(-1)  # Default to no-op

        # Iterate through environments (parallel processing of environments)
        for batch in range(self.parallel_envs):
            max_score = -100
            max_fire_index = -1
            # Consider each fire in the environment
            for fire_idx in range(self.observation['tasks'].size(1)):
                fire_x = self.observation['tasks'][batch, fire_idx, 1]
                fire_y = self.observation['tasks'][batch, fire_idx, 0]
                fire_level = self.observation['tasks'][batch, fire_idx, 2]
                intensity = self.observation['tasks'][batch, fire_idx, 3]

                # Calculate the distance from agent to fire
                distance = torch.sqrt((fire_x - self_x[batch]) ** 2 + (fire_y - self_y[batch]) ** 2)
                # Calculate the suppression power required (intensity is a key factor)
                suppression_needed = min(fire_level * intensity, self_fire_power[batch])

                # Evaluate the score for fighting the fire
                if self_suppressant[batch] > 0 and suppression_needed <= self_suppressant[batch]:
                    # Higher score for more intense and closer fires
                    score = (2 / ((0.1 * distance + 1) * (0.2 * intensity + 4)))
                    if score > max_score:
                        max_score = score
                        max_fire_index = fire_idx

            if max_fire_index != -1:
                # Assign the action to fight the most dangerous fire
                self.actions[batch, 0] = max_fire_index
                self.actions[batch, 1] = 0  # Action for suppression

            # If no suitable fire to fight, noop
            if max_fire_index == -1:
                self.actions[batch].fill_(-1)

        # Mask actions for agents with no suppressant
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
        
        # Extract relevant information from the environment
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3, 2]]  # (ypos, xpos, intensity, fire_level)
        self.argmax_store = torch.zeros_like(self.fires)

        for batch in range(self.parallel_envs):
            for fire_idx in range(self.fires[batch].size(0)):
                self.argmax_store[batch][fire_idx] = self.fires[batch][fire_idx]
