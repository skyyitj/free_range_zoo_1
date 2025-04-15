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
        super().__init__(*args, **kwargs)
        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def observe(self, observation: Dict[str, Any]) -> None:
        try:
            self.observation, self.t_mapping = observation
            self.t_mapping = self.t_mapping['agent_action_mapping']
        except Exception:
            self.observation = observation

        self.self_obs = self.observation['self']
        self.fire_tasks = self.observation['tasks'].to_padded_tensor(-100)  # [envs, num_tasks, 4]
        self.fire_scores = torch.zeros_like(self.fire_tasks[:, :, 0])  # score per task per env

        for env in range(self.parallel_envs):
            agent_y, agent_x = self.self_obs[env, 0], self.self_obs[env, 1]
            suppressant = self.self_obs[env, 3]

            for i, task in enumerate(self.fire_tasks[env]):
                y, x, _, intensity = task
                if y < 0 or x < 0:  # Padding
                    continue
                dist = torch.abs(agent_y - y) + torch.abs(agent_x - x)
                # Score fires: prioritize high intensity + close proximity
                score = (intensity + 1) / (dist + 1)
                # Penalize if agent lacks suppressant
                if suppressant == 0:
                    score *= 0.1
                self.fire_scores[env, i] = score

        # Track best task per env
        self.best_tasks = torch.argmax(self.fire_scores, dim=1)
        self.no_valid_fire = torch.all(self.fire_scores <= 0.1, dim=1)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        has_suppressant = self.self_obs[:, 3] > 0
        for env in range(self.parallel_envs):
            if self.no_valid_fire[env]:
                self.actions[env].fill_(-1)
                continue

            task_index = self.best_tasks[env].item()
            if [task_index, 0] in action_space.spaces[env].enumerate():
                self.actions[env, 0] = task_index
                self.actions[env, 1] = 0 if has_suppressant[env] else -1
            else:
                self.actions[env].fill_(-1)
        return self.actions.tolist()
