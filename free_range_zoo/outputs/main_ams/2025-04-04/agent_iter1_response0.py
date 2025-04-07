# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free_range_zoo.utils.agent import Agent
class GenerateAgent(ABC):
    """A policy agent for wildfire suppression."""

    def __init__(self, agent_name: str, parallel_envs: int, num_tasks: int) -> None:
        """
        Initialize the agent.

        Args:
            agent_name: str - Name of the subject agent.
            parallel_envs: int - Number of parallel environments to operate on.
            num_tasks: int - Number of fire suppression tasks available.
        """
        self.agent_name = agent_name
        self.parallel_envs = parallel_envs
        self.num_tasks = num_tasks
        self.action_space = build_action_space(num_tasks)
        # Internal state for observation storage (to be updated during `observe` calls)
        self.observation = None

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observation = observation

    def act(self, action_space: Any) -> List[List[int]]:
        """
        Select actions for suppression based on observed state.

        Args:
            action_space: Any - Action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        actions = []

        # Iterate through parallel environments to decide actions
        for env_id in range(self.parallel_envs):
            obs = self.observation
            
            # Extract relevant sub-components of current observation
            fires = obs['tasks']  # (fire_position_y, fire_position_x, fire_level, fire_intensity)
            agents = obs['self']  # Agent's own state: position, suppressant remaining, etc.

            selected_action = self.select_optimal_action(fires, agents, action_space)
            actions.append(selected_action)

        return actions

    def select_optimal_action(self, fires: List[np.ndarray], agents: np.ndarray, action_space: Any) -> List[int]:
        """
        Use a heuristic-based approach to select the optimal action for the agent.

        Args:
            fires: List[np.ndarray] - Observations about fires (position, level, intensity, etc.).
            agents: np.ndarray - Agent's current state: position, suppressant, etc.
            action_space: Any - Available actions for the agent to take.

        Returns:
            List[int] - Chosen action for the agent.
        """
        agent_position = agents[:2]  # Extract agent's position
        suppressant_level = agents[2]  # Extract remaining suppressant level

        # If the agent lacks suppressant or proper equipment, skip the turn (-1 action ID)
        if suppressant_level <= 0:
            return [-1, -1]  # No operation

        # Calculate distances to fires and prioritize targets
        fire_scores = []
        for fire_id, fire in enumerate(fires):
            fire_position = fire[:2]
            fire_level = fire[2]
            fire_intensity = fire[3]
            distance = np.linalg.norm(agent_position - fire_position)

            # Score fires based on proximity, intensity, and remaining fuel (higher is better)
            score = (
                (5 - distance) * 1.5 +  # Proximity weighting
                fire_intensity * 2.0 +  # Intensity weighting
                fire_level * 1.0       # Fire level (remaining fuel proxy)
            )
            fire_scores.append((score, fire_id))

        # Sort fires by descending score
        fire_scores.sort(reverse=True, key=lambda x: x[0])

        # Select the highest priority fire within the attack range
        for score, fire_id in fire_scores:
            fire = fires[fire_id]
            fire_position = fire[:2]
            distance = np.linalg.norm(agent_position - fire_position)

            if distance <= 3.0:  # If the fire is within the attack range
                return [fire_id, 0]  # Choose to attack this fire using suppressant (action ID 0)

        # If no valid target exists, skip the turn
        return [-1, -1]
