# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free_range_z oo.utils.agent import Agent
class GenerateAgent:
    """Agent that prioritizes extinguishing the most intense wildfires within its capabilities."""

    def __init__(self, max_power: float, max_suppressant: float, include_power: bool, include_suppressant: bool) -> None:
        """Initialize the agent with necessary configuration."""
        self.max_power = max_power
        self.max_suppressant = max_suppressant
        self.include_power = include_power
        self.include_suppressant = include_suppressant
        self.current_observation = {}

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment and process the observations.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.current_observation = observation

    def act(self, action_space) -> List[List[int]]:
        """
        Decide actions based on current observations.

        Args:
            action_space: Action space available to the agent.

        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []

        # Retrieve relevant observation data
        self_obs = self.current_observation.get('self')
        tasks_obs = self.current_observation.get('tasks')

        # Base decision strategy on the energy level, suppressant level, and tasks
        for env_index in range(len(self_obs)):
            agent_position = self_obs[env_index][:2]
            agent_energy = self_obs[env_index][2] if self.include_power else 1.0
            agent_suppressant = self_obs[env_index][3] if self.include_suppressant else 1.0

            # Prioritize fires by intensity and proximity
            selected_action = None
            max_priority = -1

            for i, fire in enumerate(tasks_obs):
                fire_position = fire[:2]
                fire_intensity = fire[3]

                # Compute a simple priority score
                distance = np.linalg.norm(np.array(fire_position) - np.array(agent_position))
                priority = fire_intensity / max(distance, 0.1)  # Prevent division by zero

                if priority > max_priority and fire_intensity <= agent_energy * agent_suppressant:
                    max_priority = priority
                    selected_action = i

            actions.append([selected_action if selected_action is not None else -1])  # -1 for no valid action

        return actions
