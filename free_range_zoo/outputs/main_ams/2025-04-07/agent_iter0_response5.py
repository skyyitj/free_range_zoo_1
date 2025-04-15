# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class WildfireFightingAgent(Agent):
    """Agent that coordinates firefighting efforts with resource management and dynamic fire suppression."""

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
        self_x = self.observation['self'][:, 1]
        self_y = self.observation['self'][:, 0]
        self_suppressant = self.observation['self'][:, 3]
        self_fire_power = self.observation['self'][:, 2]

        # Create action storage
        for batch in range(self.parallel_envs):
            max_score = -float('inf')
            best_action = -1
            target_fire = None

            # Check all possible fire positions and calculate the suppression score
            for fire_idx, fire in enumerate(self.observation['tasks']):
                fire_x = fire[1]
                fire_y = fire[0]
                fire_intensity = fire[2]
                fire_level = fire[3]
                distance = torch.sqrt((self_x[batch] - fire_x) ** 2 + (self_y[batch] - fire_y) ** 2)
                
                # Prioritize fires that have high intensity and are close by
                suppression_score = self._compute_suppression_score(distance, fire_intensity, self_suppressant[batch])
                
                if suppression_score > max_score:
                    max_score = suppression_score
                    target_fire = fire_idx

            # If no valid fire is found or agent has no suppressant, noop
            if target_fire is None or self_suppressant[batch] == 0:
                self.actions[batch].fill_(-1)
            else:
                # Assign action to target the best fire
                self.actions[batch, 0] = target_fire
                self.actions[batch, 1] = 0  # Action type (e.g., 'fight')

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
        except Exception as e:  # Exception handling for any observation structure issues
            self.observation = observation

        # Process fires and store relevant data for fire suppression
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 2, 3]]  # (y, x, intensity, fire_level)
        self.argmax_store = torch.zeros_like(self.fires)

        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]

    def _compute_suppression_score(self, distance: torch.Tensor, intensity: torch.Tensor, suppressant: torch.Tensor) -> torch.Tensor:
        """
        Compute the suppression score for a given fire based on distance, intensity, and available suppressant.

        Args:
            distance: Tensor - Distance from the agent to the fire.
            intensity: Tensor - Intensity of the fire at a given location.
            suppressant: Tensor - Amount of suppressant available for the agent.

        Returns:
            score: Tensor - Suppression score for the fire.
        """
        alpha = 1.0  # Distance weight factor
        beta = 0.5  # Intensity weight factor
        gamma = 0.1  # Suppressant weight factor

        # Compute suppression effectiveness based on distance, intensity, and suppressant availability
        score = (2 / ((alpha * distance + 1) * (beta * intensity + 1))) * (gamma * suppressant + 1)
        return score
