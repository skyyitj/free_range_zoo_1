from typing import Tuple, List
import numpy as np

def single_agent_policy(
    # Agent Properties
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # Team Information
    other_agents_pos: List[Tuple[float, float]], 

    # Fire Task Information
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                    
    fire_intensities: List[float],              

    # Task Prioritization
    fire_putout_weight: List[float],             
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """

    best_score = -np.inf  
    best_task = -1 

    # Parameters for the exponential scoring approach
    distance_temp = 0.5  
    intensity_temp = 1.0  
    suppressant_temp = 0.3 

    for task_index in range(len(fire_pos)):
        # Compute the Euclidean distance to the fire
        distance = np.linalg.norm(np.subtract(agent_pos, fire_pos[task_index]))

        # Compute the potential extinguishing power of the agent for this fire
        potential_extinguishing_power = min(agent_suppressant_num * agent_fire_reduction_power,
                                            fire_intensities[task_index])

        # Compute task scores
        distance_score = np.exp(-distance_temp * distance)
        intensity_score = np.exp(intensity_temp * potential_extinguishing_power)
        suppressant_score = np.exp(-suppressant_temp * agent_suppressant_num)

        # Compute total score for the task
        total_score = fire_putout_weight[task_index] * (distance_score + intensity_score + suppressant_score)

        # Update best task if total score is higher
        if total_score > best_score:
            best_score = total_score
            best_task = task_index

    return best_task