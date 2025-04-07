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
    """Agent that coordinates multiple firefighters to suppress wildfires."""

    def __init__(self, *args, **kwargs) -> None:
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
        self_firepower = self.observation['self'][:, 2]
        
        actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)
        
        for batch in range(self.parallel_envs):
            fire_data = self.observation['tasks'][batch]  # Shape: [num_fires, 4] (ypos, xpos, fire_level, intensity)
            best_action = -1
            max_score = -float('inf')

            if fire_data.size(0) == 0:  # No fires in the environment
                actions[batch].fill_(-1)
                continue

            for fire_idx in range(fire_data.size(0)):
                fire_x = fire_data[fire_idx, 1]
                fire_y = fire_data[fire_idx, 0]
                fire_level = fire_data[fire_idx, 2]
                intensity = fire_data[fire_idx, 3]
                
                # Calculate distance to the fire
                distance = torch.sqrt((fire_x - self_x[batch])**2 + (fire_y - self_y[batch])**2)

                # Determine score based on the fire's intensity and distance from the agent
                score = self.calculate_firefighter_score(fire_x, fire_y, fire_level, intensity, distance[batch], self_suppressant[batch], self_firepower[batch])
                
                if score > max_score:
                    max_score = score
                    best_action = fire_idx

            if best_action == -1:  # No valid action found
                actions[batch].fill_(-1)
            else:
                actions[batch, 0] = best_action  # Select fire task to act upon
                actions[batch, 1] = 0  # Fighting action (not no-op)

        # Apply suppressant availability condition: if no suppressant, agent does nothing
        actions[:, 1].masked_fill_(self_suppressant == 0, -1)

        return actions

    def calculate_firefighter_score(self, fire_x, fire_y, fire_level, intensity, distance, suppressant, firepower):
        """
        Calculate a score for each fire based on distance, fire intensity, and available resources.
        Args:
            fire_x (int): x-coordinate of the fire
            fire_y (int): y-coordinate of the fire
            fire_level (int): Fire level
            intensity (int): Fire intensity
            distance (float): Distance to the fire
            suppressant (float): Available suppressant
            firepower (float): Available firepower

        Returns:
            score (float): Calculated score for the fire
        """
        # Parameters that can be adjusted to weight the factors
        alpha = 1.0  # Distance weight
        beta = 0.5   # Intensity weight
        gamma = 0.2  # Firepower weight
        delta = 0.3  # Suppressant weight
        
        # Calculate score based on a function of distance, fire intensity, suppressant, and firepower
        distance_score = 1 / (alpha * distance + 1)  # Inverse distance score, closer is better
        intensity_score = intensity / (beta * fire_level + 1)  # Fire intensity; more intense fires are more critical
        firepower_score = firepower / (gamma * fire_level + 1)  # Higher firepower increases the score
        suppressant_score = suppressant / (delta * fire_level + 1)  # Higher suppressant improves score

        # Total score is a weighted combination of all the factors
        score = distance_score + intensity_score + firepower_score + suppressant_score
        return score

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

        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]  # [y, x, intensity]
        self.argmax_store = torch.zeros_like(self.fires)

        # Store fire data for each environment in batch
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
