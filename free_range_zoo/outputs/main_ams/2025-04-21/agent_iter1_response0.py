import math
import numpy as np
from typing import List, Tuple

def revised_single_agent_policy(
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                      
    fire_intensities: List[float],                

    fire_putout_weight: List[float],             
) -> int:
    
    num_fires = len(fire_pos)
    num_agents = len(other_agents_pos) + 1  # including the current agent

    # Temperature variables for scoring transformations
    distance_temp = 0.1
    intensity_temp = 0.5
    weight_temp = 2.0
    
    task_scores = []
    for i in range(num_fires):

        # Calculate Euclidean distance to the fire
        distance = math.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        # Normalize distance using exponential function
        exp_distance = np.exp(-distance / distance_temp)
        
        # Normalize fire intensity
        fire_intensity = fire_intensities[i]
        exp_intensity = np.exp(-fire_intensity / intensity_temp)

        # Consider fire's priority weight
        task_weight = fire_putout_weight[i]
        exp_weight = np.exp(task_weight / weight_temp)
        
        # Calculate potential fire suppression rate on this task
        potential_suppression = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        
        # Calculate task score and give penalty if the fire is assigned to more than one agent
        agent_to_fire_ratio = num_agents / num_fires
        task_score = (exp_distance * exp_intensity * exp_weight * potential_suppression) / agent_to_fire_ratio  
        task_scores.append(task_score)
        
    # Return the index of the fire task with maximum score
    return task_scores.index(max(task_scores))