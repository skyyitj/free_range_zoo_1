# Auto-generated Agent Class
from typing import Tuple, List, Dict, Any
import torch
import free_range_rust
from torch import Tensor
from free_range_rust import Space
import numpy as np
from collections import defaultdict
from free_range_zoo.utils.agent import Agent
class GenerateAgent:
    def __init__(self, agent_name, observation_space, action_space):
        self.agent_name = agent_name
        self.observation_space = observation_space
        self.action_space = action_space
        self.last_observation = None

    def observe(self, observation):
        """
        Normalize and parse the observation vector into meaningful components.
        Assume the observation includes:
        - fire map (flattened),
        - agent position,
        - water level,
        - and critical area threat levels.
        """
        self.last_observation = observation

        fire_map_size = 100  # e.g. 10x10 grid
        fire_map = observation[:fire_map_size].reshape(10, 10)
        agent_pos = observation[fire_map_size:fire_map_size+2]
        water_level = observation[fire_map_size+2]
        critical_area_threat = observation[fire_map_size+3:fire_map_size+6]

        return {
            "fire_map": fire_map,
            "agent_pos": agent_pos,
            "water_level": water_level,
            "critical_threat": critical_area_threat
        }

    def act(self):
        """
        Use heuristics to choose action:
        - Move towards highest fire intensity.
        - Prioritize suppression if water is available.
        - Move to refill zone if water low.
        """
        obs = self.observe(self.last_observation)
        fire_map = obs["fire_map"]
        agent_x, agent_y = map(int, obs["agent_pos"])
        water_level = obs["water_level"]

        if water_level < 0.1:
            # Refill strategy: head to bottom-left (0,0)
            if agent_x > 0:
                return 0  # Move left
            elif agent_y > 0:
                return 1  # Move down
            else:
                return 4  # Refill action
        else:
            # Suppression strategy: find nearby highest fire
            max_fire = -1
            target = (agent_x, agent_y)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = agent_x + dx, agent_y + dy
                    if 0 <= nx < 10 and 0 <= ny < 10:
                        if fire_map[nx][ny] > max_fire:
                            max_fire = fire_map[nx][ny]
                            target = (nx, ny)

            # Choose direction toward target
            tx, ty = target
            if tx > agent_x:
                return 2  # Right
            elif tx < agent_x:
                return 0  # Left
            elif ty > agent_y:
                return 3  # Up
            elif ty < agent_y:
                return 1  # Down
            else:
                return 5  # Suppress fire
