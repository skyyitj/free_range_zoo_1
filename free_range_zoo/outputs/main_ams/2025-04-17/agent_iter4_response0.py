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
    Optimal fire-fighting task selection for a single agent considering fire intensity, distance, and agent capabilities.
    """

    # Initialize variables
    best_score = -np.inf  
    best_task = -1 

    if agent_suppressant_num <= 0:
        return best_task  # Agent has no suppressants.

    for task_idx in range(len(fire_pos)):
        # Calculate the Euclidean distance to the fire
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[task_idx]))
        
        # Consider if the agent can make a significant impact on the fire
        effectiveness = agent_fire_reduction_power / (1 + fire_intensities[task_idx])
        
        # Create a score blend of priority, distance, and potential effectiveness
        score = (fire_putout_weight[task_idx] * effectiveness) / (1 + distance)  
        
        # Update best task if new score is higher
        if score > best_score:
            best_score = score
            best_task = task_idx

    return best_task