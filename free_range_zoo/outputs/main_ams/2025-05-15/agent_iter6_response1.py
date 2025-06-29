import numpy as np
from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_supressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1
    
    # Tuning these parameters based on metric results
    distance_temperature = 0.002  # Further precision-focused
    intensity_temperature = 0.01  # Higher sensitivity to intensity changes
    suppressant_efficiency_scale = 30.0  # More conservation of suppressant 
    reduction_power_scale = 2  # More emphasis on the agent's fire reduction power
    
    suppressant_remain_scale = np.exp(-suppressant_efficiency_scale * (1 - (agent_suppressant_num / 10)))
    
    for i in range(num_tasks):
        (fy, fx) = fire_pos[i]
        distance = np.sqrt((agent_pos[0] - fy)**2 + (agent_pos[1] - fx)**2)
        norm_distance = np.exp(-distance_temperature * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)
        
        score = (
            fire_putout_weight[i] *  # Emphasis on the reward of the task
            norm_distance *  # Nearby fires prioritized
            (agent_fire_reduction_power * reduction_power_scale / (1 + intensity)) *  # Effectiveness of agent
            np.log1p(fire_levels[i]) *  # Non-linear benefits from reducing higher levels
            suppressant_remain_scale *  # Preservation of suppressants
            norm_intensity  # Effective on handling intense fires
        )
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index