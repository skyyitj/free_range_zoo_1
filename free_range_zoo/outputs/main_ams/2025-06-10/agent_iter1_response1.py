import numpy as np
from typing import List, Tuple

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
    
    # Parameters and weights for the scoring function
    distance_weight = -1.0
    intensity_weight = 1.0
    reduction_power_temperature = 5.0
    suppressant_temperature = 3.0
    weight_temperature = 2.0
    
    best_task = -1
    best_score = -float('inf')
    
    # Calculate scores for each task
    for task_index in range(num_tasks):
        # Distance from agent to fire
        fire_y, fire_x = fire_pos[task_index][0], fire_pos[task_index][1]
        distance = np.sqrt((agent_pos[0] - fire_y) ** 2 + (agent_pos[1] - fire_x) ** 2)
        normalized_distance = np.exp(distance_weight * distance)
        
        # Fire suppression potential (considering fire levels and agent's capability)
        fire_suppression_potential = (
            agent_fire_reduction_power * agent_suppressant_num / (fire_levels[task_index] * fire_intensities[task_index] + 1)
        )
        fire_suppression_potential = np.exp(reduction_power_temperature * fire_suppression_potential)
        
        # Effective suppressability factor (under current suppressant constraints)
        effective_suppressability = np.exp(suppressant_temperature * min(agent_suppressant_num, 1))
        
        # Weighted priority for putting out the fire
        fire_weight = np.exp(weight_temperature * fire_putout_weight[task_index])
        
        # Combining the factors to calculate the overall task score
        score = (normalized_distance * fire_suppression_potential * effective_suppressability * fire_weight)
        
        # Update the best task based on the highest score
        if score > best_score:
            best_score = score
            best_task = task_index
    
    return best_task