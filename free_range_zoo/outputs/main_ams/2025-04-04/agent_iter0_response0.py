# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free_range_zoo.utils.agent import Agent
class GenerateAgent(Env.Agent):
    """
    Agent for extinguishing wildfires.
    """

    def __init__(self, agent_name: str, parallel_envs: int, fire_config: FireConfiguration, agent_config: AgentConfiguration) -> None:
        """
        Initialize the agent.
        
        Args:
            agent_name: str - Name of the agent
            parallel_envs: int - Number of parallel environments
            fire_config: FireConfiguration - Configuration for wildfire behavior
            agent_config: AgentConfiguration - Configuration for agent behavior
        """
        super().__init__(agent_name, parallel_envs)
        self.fire_config = fire_config
        self.agent_config = agent_config
        self.current_observation = None

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.current_observation = observation

    def act(self, action_space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: gymnasium.spaces.MultiDiscrete - Current action space available to agents.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []

        for env_id in range(self.parallel_envs):
            # Extract observations for this environment
            fires = self.current_observation[env_id]['tasks']  # List of fires (y, x, level, intensity)
            agents = self.current_observation[env_id]['self']  # Agent state (y, x, suppressants, equipment state)

            agent_pos = np.array(agents[:2])  # (y, x) position of the agent
            suppressant_count = agents[2]    # Remaining suppressant
            equipment_state = agents[3]      # Equipment state

            # If no suppressant is left or equipment is broken, skip action
            if suppressant_count <= 0 or equipment_state == 0:
                actions.append([-1])  # No operation
                continue

            # Prioritize fires by intensity and proximity
            prioritized_fires = sorted(fires, key=lambda f: (-f[3], np.linalg.norm(agent_pos - np.array(f[:2]))))

            # Select the most critical fire within attack range
            action_chosen = [-1]  # Default to no operation

            for fire in prioritized_fires:
                fire_pos = np.array(fire[:2])  # (y, x) position of the fire
                fire_intensity = fire[3]  # Intensity of the fire

                # Compute distance to the fire
                distance = np.linalg.norm(agent_pos - fire_pos)

                # Check if the fire is within attack range
                if distance <= self.agent_config.attack_range:
                    # Attack this fire
                    action_chosen = [int(fire[2]), 0]  # Target fire index, action type: use suppressant
                    break

            actions.append(action_chosen)

        return actions
