# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

class GenerateAgent:
    """Agent that prioritizes fighting the strongest fire with the highest risk."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        self.current_observation = None

    def act(self, action_space: torch.Tensor) -> List[List[int]]:
        """
        Decide actions based on the current observation.
        
        Args:
            action_space: torch.Tensor - Current action space available to the agent.
        Returns:
            List[List[int]]: List of actions, one for each parallel environment.
        """
        actions = []
        for env_index, observation in enumerate(self.current_observation):
            # Extract useful state variables
            fires = observation["fires"]  # Fire existence matrix (0 = no fire, >0 = fire)
            intensities = observation["intensity"]  # Fire intensity matrix
            fuel = observation["fuel"]  # Remaining fuel matrix
            agent_positions = observation["agents"]  # Agent positions [(y, x)]
            suppressants = observation["suppressants"]  # Remaining suppressants per agent
            capacities = observation["capacity"]  # Agent suppressant capacities

            # Iterate over each agent
            agent_actions = []
            for agent_id, agent_position in enumerate(agent_positions):
                if suppressants[env_index, agent_id] <= 0:
                    # Skip if no suppressant is left for this agent
                    agent_actions.append([-1, -1])  # No operation
                    continue

                # Calculate fire priorities: (high intensity * low fuel) fires are prioritized
                fire_priority = (
                    (fires[env_index] > 0).float() * intensities[env_index] / (fuel[env_index] + 0.1)
                )
                # Find the highest priority target within attack range
                best_target = None
                best_score = -float("inf")
                agent_y, agent_x = agent_position
                
                for fire_y in range(fire_priority.shape[0]):
                    for fire_x in range(fire_priority.shape[1]):
                        distance = abs(agent_y - fire_y) + abs(agent_x - fire_x)
                        if distance <= 1 and fire_priority[fire_y, fire_x] > best_score:
                            best_score = fire_priority[fire_y, fire_x]
                            best_target = (fire_y, fire_x)

                # Decide action for this agent
                if best_target is not None:
                    fire_y, fire_x = best_target
                    attack_type = 0  # Use suppressant
                    agent_actions.append([fire_y * fire_priority.shape[1] + fire_x, attack_type])  # Flatten index
                else:
                    agent_actions.append([-1, -1])  # No valid target found; agent does nothing

            actions.append(agent_actions)

        return actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.
        
        Args:
            observation (Dict[str, Any]): Current observation of the environment.
        """
        self.current_observation = observation
