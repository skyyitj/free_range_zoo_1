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
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.last_observation = None

    def observe(self, observation):
        self.last_observation = observation
        # Normalize and pre-process observation for better action selection
        self.fire_map = np.array(observation["fire_map"])
        self.agent_position = observation["agent_position"]
        self.wind_direction = observation.get("wind_direction", (0, 0))  # Optional field
        self.resources = observation.get("resources", 1.0)

    def act(self):
        # Heuristic policy: prioritize extinguishing fire, then move toward it
        actions = self.action_space
        x, y = self.agent_position
        height, width = self.fire_map.shape

        def in_bounds(pos):
            return 0 <= pos[0] < height and 0 <= pos[1] < width

        # Define directions
        directions = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }

        # Step 1: Try to extinguish current location
        if self.fire_map[x][y] == 1:
            return "EXTINGUISH"

        # Step 2: Move toward nearest fire
        fire_locations = np.argwhere(self.fire_map == 1)
        if len(fire_locations) > 0:
            distances = [np.linalg.norm([fx - x, fy - y]) for fx, fy in fire_locations]
            target = fire_locations[np.argmin(distances)]
            dx, dy = target[0] - x, target[1] - y

            # Move vertically if needed
            if abs(dx) > abs(dy):
                step = ("DOWN" if dx > 0 else "UP")
            else:
                step = ("RIGHT" if dy > 0 else "LEFT")

            new_x, new_y = x + directions[step][0], y + directions[step][1]
            if in_bounds((new_x, new_y)):
                return step

        # Step 3: No fire found or can't move — do nothing
        return "STAY"
