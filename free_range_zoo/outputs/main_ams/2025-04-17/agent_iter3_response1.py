from typing import List, Tuple
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
    This function selects the optimal fire-fighting task for a single agent.
    """

    # Set an initial minimum distance to inf
    min_distance = np.inf
    # Placeholder for the optimal task
    optimal_task = -1 

    for task_idx in range(len(fire_pos)):
        
        # Calculate Euclidean distance from agent to each fire
        distance = np.sqrt((agent_pos[0] - fire_pos[task_idx][0])**2 + (agent_pos[1] - fire_pos[task_idx][1])**2)
        
        # Check if the agent has enough suppressant to put out the fire completely or reduce its intensity significantly
        enough_suppressants = agent_suppressant_num * agent_fire_reduction_power >= fire_intensities[task_idx]

        # Prioritize tasks with higher weights and which the agent has enough suppressants to handle.
        if fire_putout_weight[task_idx] > min_distance and enough_suppressants:
            min_distance = fire_putout_weight[task_idx]
            optimal_task = task_idx

    return optimal_task