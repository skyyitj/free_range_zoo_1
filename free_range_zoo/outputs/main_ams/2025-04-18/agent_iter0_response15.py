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
    # Score components temperature parameters
    distance_temp = 0.1  # Temperature parameter for the distance score component
    intensity_temp = 0.05  # Temperature parameter for the intensity score component
    weight_temp = 0.2  # Temperature parameter for the weight score component
    
    scores = []
    for i, (f_pos, f_level, f_intensity, f_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate distance between agent and fire location
        distance = np.sqrt((agent_pos[0] - f_pos[0]) ** 2 + (agent_pos[1] - f_pos[1]) ** 2)
        
        # Calculate normalized distance score (inverse, the closer the better)
        distance_score = np.exp(-distance * distance_temp)
        
        # Calculate effectiveness of agent on the fire based on its intensity
        effectiveness_score = np.exp(-f_intensity * agent_fire_reduction_power * intensity_temp)
        
        # Incorporate the strategic weight of the fire
        weight_score = np.exp(f_weight * weight_temp)
        
        # Calculate combined score for prioritizing tasks
        combined_score = (distance_score * effectiveness_score * weight_score)
        
        # Append the score to the list
        scores.append(combined_score)
    
    # Select the task with the highest score
    selected_task = np.argmax(scores)

    return selected_task