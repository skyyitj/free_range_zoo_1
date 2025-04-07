# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

from free range_ zoo utils agent import Agent
class GenerateAgent:
    """Agent that always fights the strongest available fire."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the agent."""
        self.observation = None

    def act(self, action_space) -> list(list[int]):
        """
        Return a list of actions, one for each parallel environment.
        Args:
            action_space: The current action space available to the agent.
        Returns:
            List[List[int]]: List of actions, one for each parallel environment.
        """
        
        actions = []
        
        for agent_obs in self.observation['self']:
            # Extract agent's state
            agent_pos = agent_obs[:2]
            agent_power = agent_obs[2] if len(agent_obs) > 2 else None
            agent_suppressant = agent_obs[3] if len(agent_obs) > 3 else None

            strongest_fire = None
            max_intensity = -1

            # Evaluate tasks to find the strongest fire
            for fire_obs in self.observation['tasks']:
                fire_intensity = fire_obs[3]
                
                if fire_intensity > max_intensity:
                    max_intensity = fire_intensity
                    strongest_fire = fire_obs

            # Determine action based on fire characteristics and agent state
            if strongest_fire:
                fire_position = strongest_fire[:2]
                action = self.decide_action_based_on_fire(agent_pos, fire_position, agent_power, agent_suppressant)
            else:
                action = 0  # Default action when no fire

            actions.append(action)

        return actions

    def observe(self, observation: dict) -> None:
        """
        Observe the environment.
        Args:
            observation: Current observation from the environment.
        """
        self.observation = observation

    def decide_action_based_on_fire(self, agent_pos, fire_position, agent_power, agent_suppressant) -> int:
        """
        Decide action based on the position of the agent and the fire.
        Args:
            agent_pos: Position of the agent.
            fire_position: Position of the fire.
            agent_power: Power level of the agent.
            agent_suppressant: Suppressant level of the agent.
        Returns:
            int: Action to take (index).
        """
        
        # Compute distance to fire
        distance = ((agent_pos[0] - fire_position[0]) ** 2 + (agent_pos[1] - fire_position[1]) ** 2) ** 0.5

        if agent_power is not None and agent_power < 0.3:
            return 1  # Retreat or recharge if power is low

        if agent_suppressant is not None and agent_suppressant > 0.2:
            return 2  # Attack fire

        # If nearby and resources are sufficient, move to the strongest fire
        if distance < 5:
            return 3  # Move to fire

        return 4  # Default action if none of the conditions are met
