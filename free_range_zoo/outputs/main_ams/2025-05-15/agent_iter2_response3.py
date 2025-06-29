import numpy as np
from typing import List, Tuple

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
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Modify normalization temperatures based on evaluation metrics feedback
    distance_normalization_temp = 0.3
    intensity_normalization_temp = 0.05
    level_normalization_temp = 0.2
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        # Using agent's fire reduction capability vs. the intensity of the fire
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        
        # Added to consider the current fire level at each task
        norm_level = np.exp(-level_normalization_temp * fire_levels[i])
        
        # Enhanced score calculation considering the relative capability of the agent
        score = fire_putout_weight[i] * (agent_fire_reduction_power / (1.0 + intensity)) * norm_distance * norm_intensity * norm_level
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index