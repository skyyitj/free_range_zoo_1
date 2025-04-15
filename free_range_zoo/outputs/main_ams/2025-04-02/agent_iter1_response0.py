# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free range_ zoo utils agent import Agent
class GenerateAgent(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        self.state = None

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.state = observation

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []
        # For each parallel environment
        for i in range(action_space.n):
            # To determine distance from each fire
            distances = []
            # Fire details
            fire_level = []
            fire_intensity = []
            for task in self.state['tasks'][i]:
                distances.append(np.sqrt((self.state['self'][i][0] - task[0]) ** 2 + (self.state['self'][i][1] - task[1]) ** 2))
                fire_level.append(task[2])
                fire_intensity.append(task[3])
        
            # Index of closest active fire
            closest_fire_idx = np.argmin(distances)
            if self.state['self'][i][2] > fire_level[closest_fire_idx] and self.state['self'][i][3] > fire_intensity[closest_fire_idx]:
                # Able to fight the fire
                actions.append([closest_fire_idx])
            else:
                # Need to recharge
                actions.append(['Recharge'])
        
        return actions
