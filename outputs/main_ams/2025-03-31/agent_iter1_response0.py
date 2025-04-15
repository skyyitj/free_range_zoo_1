# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

class GenerateAgent:
    """Agent optimized to fight fires based on priority targeting."""

    def __init__(self, agent_id: int, env_config: Dict[str, Any]) -> None:
        """
        Initialize the agent.
        Args:
            agent_id: Unique ID for this agent.
            env_config: Environment configuration containing fire, agent, and reward settings.
        """
        self.agent_id = agent_id
        self.env_config = env_config
        self.reward_multiplier = env_config['reward_scale']

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment (state representation).
        Args:
            observation: Environment observation containing fire and agent states.
        """
        self.observation = observation

    def act(self, action_space: List[List[int]]) -> List[int]:
        """
        Return the most optimal action for the current environment state.
        Args:
            action_space: Enumerated valid actions for the agent.
        Returns:
            List[int]: The optimal action based on current state and priority logic.
        """
        # Parse observation into actionable details
        fires = self.observation['tasks']  # List of fire states
        agent_state = self.observation['self']  # Agent-specific observation

        agent_position = np.array(agent_state[:2])  # Agent's (y, x) position
        agent_capacity = agent_state[4]  # Remaining suppressant capacity
        active_suppressant = agent_state[3]  # Active suppressant level

        prioritized_target = None
        min_distance = float('inf')

        # Iterate over fires to prioritize targets based on intensity, proximity, and resource usage
        for fire_obs in fires:
            fire_position = np.array(fire_obs[:2])
            fire_level = fire_obs[2]
            fire_intensity = fire_obs[3]

            distance = np.linalg.norm(agent_position - fire_position)

            # Only select reachable and active fires based on suppressant and attack range
            if active_suppressant > 0 and fire_intensity > 0:
                # Prioritize based on linear reward scaling with intensity
                if distance < min_distance and fire_intensity > 0:
                    prioritized_target = fire_position
                    min_distance = distance

        # Action determination: Attack prioritized fire if available
        if prioritized_target is not None:
            action_task = np.argmin(
                [np.linalg.norm(np.array(fire[:2]) - prioritized_target) for fire in fires]
            )
            action_type = 0  # 0 indicates use of suppressant
        else:
            # Default action: No operation
            action_task = -1
            action_type = -1

        # Return selected action
        return [action_task, action_type]
