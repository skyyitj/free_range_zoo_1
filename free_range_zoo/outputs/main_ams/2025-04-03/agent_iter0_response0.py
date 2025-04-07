# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free_range_z oo.utils.agent import Agent
class GenerateAgent(Agent):
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        super().__init__(*args, **kwargs)
        self.current_observation = None

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []
        if self.current_observation is not None:
            for env_idx, obs in enumerate(self.current_observation):
                fire_intensities = [task[3] for task in obs['tasks']]
                if len(fire_intensities) == 0:
                    continue
                strongest_fire = np.argmax(fire_intensities)
                action = [strongest_fire]
                actions.append(action)
        return actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.current_observation = observation
