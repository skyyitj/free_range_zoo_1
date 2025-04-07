# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class WildfireAgent(Agent):
    """Agent that coordinates firefighting efforts based on the environment's state."""
    
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
        self_x = self.observation['self'][:, 1]  # X position of the agent
        self_y = self.observation['self'][:, 0]  # Y position of the agent
        fire_intensity = self.observation['tasks'][:, :, 2]  # Intensity of the fires in each grid position
        fire_positions = self.observation['tasks'][:, :, 0:2]  # Y, X positions of the fires
        suppressant = self.observation['self'][:, 3]  # Available suppressant for the agent
        capacity = self.observation['self'][:, 2]  # Available firepower (capacity) of the agent
        
        for batch in range(self.parallel_envs):
            best_score = -math.inf
            best_action = [-1, -1]  # Default action (noop)

            # Go through each fire and decide the best action
            for fire_idx in range(fire_positions.shape[1]):
                fire_x, fire_y = fire_positions[batch, fire_idx, 1], fire_positions[batch, fire_idx, 0]
                intensity = fire_intensity[batch, fire_idx]
                
                # Calculate distance from the agent to the fire
                distance = math.sqrt((fire_x - self_x[batch])**2 + (fire_y - self_y[batch])**2)
                
                # Reward function based on distance and intensity of the fire
                fire_power = capacity[batch]  # Agent's firepower
                available_suppressant = suppressant[batch]  # Available suppressant
                
                # Compute a score based on distance and fire intensity, while considering agent's resources
                score = (intensity / (distance + 1)) + (0.5 * fire_power) + (0.2 * available_suppressant)
                
                # Update the best action if this fire provides a higher score
                if score > best_score:
                    best_score = score
                    best_action = [fire_idx, 0]  # Fight the fire (action 0)

            # If no fire is found, take a noop action (wait or idle)
            if best_score == -math.inf:
                best_action = [-1, -1]  # Noop

            self.actions[batch, 0] = best_action[0]  # Store fire index for fighting
            self.actions[batch, 1] = best_action[1]  # Store action (fight or noop)
        
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
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]
        self.argmax_store = torch.zeros_like(self.fires)

        # Process the fire data
        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]
