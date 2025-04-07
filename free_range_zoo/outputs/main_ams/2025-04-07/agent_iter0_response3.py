# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class GenerateAgent(Agent):
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
        # Observe the agent's current position, resources, and task information
        self_x = self.observation['self'][:, 1]
        self_y = self.observation['self'][:, 0]
        self_fire_power = self.observation['self'][:, 2]
        self_suppressant = self.observation['self'][:, 3]
        
        # Initialize actions for all environments
        self.actions.fill_(-1)  # Default action is 'noop' (-1)

        for batch in range(self.parallel_envs):
            # Step 1: Find the most intense fire task to fight
            max_intensity = -1
            target_fire = None
            for task in range(self.observation['tasks'].size(1)):  # Iterate over tasks (fires)
                fire_y, fire_x, fire_level, fire_intensity = self.observation['tasks'][batch][task]
                
                # Only consider fires that are not already burned out
                if fire_intensity > 0 and fire_intensity > max_intensity:
                    max_intensity = fire_intensity
                    target_fire = (fire_y, fire_x, fire_level, fire_intensity, task)

            # Step 2: If we have a fire to fight, decide action based on resources
            if target_fire:
                fire_y, fire_x, fire_level, fire_intensity, task = target_fire

                # Check if we have suppressant or power left
                if self_suppressant[batch] > 0 and self_fire_power[batch] > 0:
                    # Attack the fire if we have sufficient resources
                    self.actions[batch, 0] = task  # Select the task (fire) to fight
                    self.actions[batch, 1] = 0  # Action 0 corresponds to fighting fire
                else:
                    # If no suppressant or power, noop and wait for resources to regenerate
                    self.actions[batch].fill_(-1)
            else:
                # If no fire to fight, noop
                self.actions[batch].fill_(-1)
        
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
        except Exception as e:  # Catch all exceptions and assign observation
            self.observation = observation
        
        # Process fire task observations
        self.fires = self.observation['tasks'].to_padded_tensor(-100)[:, :, [0, 1, 3]]  # Extract fire task info
        self.argmax_store = torch.zeros_like(self.fires)

        for batch in range(self.parallel_envs):
            for element in range(self.fires[batch].size(0)):
                self.argmax_store[batch][element] = self.fires[batch][element]

        # The agent will need to consider fire location, intensity, and other agents' actions
        # To coordinate better in multi-agent settings, the agent may need to use this information to adjust its strategy
