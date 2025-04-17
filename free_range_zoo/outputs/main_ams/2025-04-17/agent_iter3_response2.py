from typing import Tuple, List
import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                    
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    # Define temperature parameters for distance, intensity and suppressant cost transformations
    distance_temp = 5 
    intensity_temp = 2 
    cost_temp = 2
    
    # Placeholder values
    optimal_task = None
    highest_score = -np.inf

    suppressant_cost = agent_suppressant_num / agent_fire_reduction_power

    for i, fire_position in enumerate(fire_pos):
        # Calculate distance, intensity and suppressant cost score
        distance_score = np.exp(-distance_temp * np.linalg.norm(np.array(agent_pos) - np.array(fire_position)))
        intensity_score = np.exp(intensity_temp * fire_intensities[i])
        suppressant_cost_score = np.exp(-cost_temp * suppressant_cost)
        
        # Combine scores according to weights
        total_score = fire_putout_weight[i] * (distance_score + intensity_score + suppressant_cost_score)

        if total_score > highest_score:
            highest_score = total_score
            optimal_task = i

    return optimal_task