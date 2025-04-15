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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_observation = None

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        actions = []

        for env_index in range(self.parallel_envs):
            current_actions = []

            for agent_index, agent_name in enumerate(self.agent_names):
                agent_obs = self.last_observation[env_index][agent_name]

                ypos, xpos, fire_power, suppressant = agent_obs['self']
                tasks = agent_obs['tasks']

                # Find the most critical task; prioritize by fire intensity
                most_critical_task = max(tasks, key=lambda x: x['intensity'])
                distance_to_task = abs(most_critical_task['y'] - ypos) + abs(most_critical_task['x'] - xpos)

                # Example decision rules
                if suppressant > 0 and distance_to_task <= 3:
                    action_to_take = tasks.index(most_critical_task)  # Accessing the index for the task
                else:
                    action_to_take = -1  # 'noop' if out of suppressant or too far

                current_actions.append(action_to_take)
            actions.append(current_actions)
        return actions

    def observe(self, observation: Dict[str, Any]) -> None:
        self.last_observation = observation
