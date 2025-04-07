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
        """
        Args:
            agent_name: str - Name of the subject agent
            parallel_envs: int - Number of parallel environments to operate on
        """
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

        actions = []

        for env_idx in range(self.parallel_envs):
            observation = self.observations[env_idx]  # Get current observation for this environment

            # Step 1: Get agent's current position and available resources
            self_position = observation['self'][:2]  # [ypos, xpos]
            fire_power = observation['self'][2]
            suppressant = observation['self'][3]

            # Step 2: Get the closest fire that needs to be suppressed
            fire_tasks = observation['tasks']
            task_to_fight = None
            max_fire_level = -1
            for fire_task in fire_tasks:
                fire_pos = fire_task[:2]  # [y, x]
                fire_level = fire_task[2]
                intensity = fire_task[3]

                # Choose the fire with the highest intensity and that is near the agent
                if self._is_adjacent(self_position, fire_pos) and fire_level > max_fire_level and intensity > 0:
                    max_fire_level = fire_level
                    task_to_fight = fire_task

            # Step 3: If a fire is found to fight, decide if enough resources are available
            if task_to_fight is not None and fire_power > 0 and suppressant > 0:
                # Fight the fire task
                actions.append([task_to_fight, 1])  # Action: Fight fire (task_to_fight)
            else:
                # If not enough resources, do nothing
                actions.append([None, -1])  # Action: No-op

        return actions

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Observe the environment.
        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """
        self.observations = observation

    def _is_adjacent(self, pos1: torch.Tensor, pos2: torch.Tensor) -> bool:
        """
        Check if two positions are adjacent (horizontally or vertically).
        Args:
            pos1: [y1, x1] - First position.
            pos2: [y2, x2] - Second position.
        Returns:
            bool: True if adjacent, False otherwise.
        """
        y_diff = abs(pos1[0] - pos2[0])
        x_diff = abs(pos1[1] - pos2[1])
        return (y_diff == 1 and x_diff == 0) or (x_diff == 1 and y_diff == 0)
