from typing import List, Tuple
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:

    # Score components transformation temperatures
    distance_temp = 20.0
    intensity_temp = 5.0
    weight_temp = 10.0
    
    best_score = -np.inf
    selected_task = 0
    
    for i, fire in enumerate(fire_pos):
        # Calculate distance from agent to fire
        distance = np.hypot(fire[0] - agent_pos[0], fire[1] - agent_pos[1])
        
        # Adjust score components with exponential to ensure more manageable ranges
        distance_score = np.exp(-distance / distance_temp)
        intensity_score = np.exp(-fire_intensities[i] / intensity_temp)
        weight_score = np.exp(fire_putout_weight[i] / weight_temp)
        
        # Overall task score, combining all aspects with specific focus
        task_score = distance_score * intensity_score * weight_score
        
        if task_score > best_score:
            best_score = task_score
            selected_task = i
    
    return selected_task