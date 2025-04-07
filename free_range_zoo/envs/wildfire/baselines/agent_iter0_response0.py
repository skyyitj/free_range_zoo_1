# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class CoordinatedFirefighterAgent(Agent):
    """Agent that coordinates with others to suppress fires while managing resources."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)  # Action storage

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.
        
        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
            
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        has_suppressant = self.observation['self'][:, 3] != 0  # Check if agent has suppressant
        self_x = self.observation['self'][:, 1]  # Agent x position
        self_y = self.observation['self'][:, 0]  # Agent y position
        
        for batch in range(self.parallel_envs):
            max_score = -float('inf')  # Initialize maximum score to a very low value
            best_action = -1  # Default to no action if no fire is detected
            
            if self.observation['tasks'][batch].size(0) == 0:  # If no fire present
                self.actions[batch].fill_(-1)  # No-op action
                continue
            
            # Iterate through each task (fire) to decide which one to prioritize
            for idx, fire in enumerate(self.observation['tasks'][batch]):
                fire_x = fire[1]  # Fire x position
                fire_y = fire[0]  # Fire y position
                fire_intensity = fire[2]  # Fire intensity

                distance = torch.sqrt((fire_x - self_x[batch]) ** 2 + (fire_y - self_y[batch]) ** 2)  # Euclidean distance
                suppression_effectiveness = (2 / (distance + 1))  # Prioritize closer fires
                suppression_effectiveness *= fire_intensity * 0.1  # Penalize fires with higher intensity

                # Calculate a score for this fire, based on distance and intensity
                score = suppression_effectiveness

                if score > max_score:
                    max_score = score
                    best_action = idx  # Select this fire to fight

            # If there's no available fire, the agent does nothing
            if best_action == -1:
                self.actions[batch].fill_(-1)
            else:
                self.actions[batch, 0] = best_action  # Set the action to fight the selected fire
                self.actions[batch, 1] = 0  # Use firepower

            # Ensure agents without suppressant do not take actions requiring it
            self.actions[batch, 1].masked_fill_(~has_suppressant[batch], -1)

        return self.actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment and process the current state.
        
        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation = observation
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]  # Fire locations and intensity
        self.argmax_store = torch.zeros_like(self.fires)  # Initialize storage for processed fire data

        # For each parallel environment, store fire data for decision-making
        for batch in range(self.parallel_envs):
            for fire_idx in range(self.fires[batch].size(0)):
                self.argmax_store[batch][fire_idx] = self.fires[batch][fire_idx]
        
        # Update any other relevant state, such as team status or fire suppression history
        self.team_capacity = self.observation['self'][:, 3]  # Get the capacity of the team members (suppressant)
