import numpy as np
from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1
    
    # Configurable temperature factors for adjusting sensitivity in scoring
    suppressant_effectiveness_temp = 0.3
    distance_temp = 0.1
    priority_weight_temp = 5.0
    intensity_weight_temp = 0.2

    # Normalize available suppressant
    normalized_suppressant = np.exp(-suppressant_effectiveness_temp * (1/agent_suppressant_num if agent_suppressant_num else float('inf')))

    for task_index in range(num_tasks):
        fire_location = fire_pos[task_index]
        fire_intensity = fire_levels[task_index] * fire_intensities[task_index]
        priority_weight = fire_putout_weight[task_index]
        
        # Calculate Euclidean distance to the fire
        distance = np.sqrt((agent_pos[0] - fire_location[0])**2 + (agent_pos[1] - fire_location[1])**2)
        
        # Calculate score components
        normalized_distance = np.exp(-distance_temp * distance)
        adjusted_priority = np.exp(priority_weight_temp * priority_weight)
        intensity_penalty = np.exp(-intensity_weight_temp * fire_intensity)
        
        # Compute the task score
        task_score = normalized_suppressant * normalized_distance * adjusted_priority * intensity_penalty
        
        # Update if found a better task
        if task_score > best_task_score:
            best_task_score = task_score
            selected_task_index = task_index

    return selected_task_index