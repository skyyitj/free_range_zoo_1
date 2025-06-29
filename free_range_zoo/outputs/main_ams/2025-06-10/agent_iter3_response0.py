import numpy as np
from typing import Tuple, List

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float]             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_pos)
    best_task_idx = -1
    best_task_score = float('-inf')

    for task_idx in range(num_tasks):
        # Calculate the distance to the fire
        dist = np.sqrt((fire_pos[task_idx][0] - agent_pos[0]) ** 2 +
                       (fire_pos[task_idx][1] - agent_pos[1]) ** 2)
        
        # Effective fire intensity after suppression would be attempted
        possible_reduction = min(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[task_idx])
        remaining_intensity = fire_intensities[task_idx] - possible_reduction
        
        # Score contribution from distance
        distance_factor = 1 / (1 + dist)  # Encourage closer targets
        # Score contribution from fire intensity reduction potential
        reduction_potential_score = possible_reduction / (1 + remaining_intensity)
        
        # Importance of the fire task
        task_importance = fire_putout_weight[task_idx]
        
        # Score for this task
        task_temperature = 0.5  # Can adjust "sharpness" of priority distinction
        score = (np.exp(reduction_potential_score / task_temperature) *
                 (task_importance * distance_factor))

        # Select the task with highest score
        if score > best_task_score:
            best_task_score = score
            best_task_idx = task_idx

    return best_task_idx