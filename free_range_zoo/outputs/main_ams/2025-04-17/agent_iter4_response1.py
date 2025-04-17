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
    Optimal fire-fighting task selection for a single agent.
    """

    # Initialize the best task index and highest score
    best_task = None
    best_score = -np.inf
    
    # Iterate over all fire tasks to evaluate them
    for i, (f_pos, f_intensity, f_weight) in enumerate(zip(fire_pos, fire_intensities, fire_putout_weight)):
        
        # Calculate various factors influencing task selection
        distance = np.linalg.norm(np.array(agent_pos) - np.array(f_pos))
        # Ensure the agent has enough suppressant to meaningfully contribute to this fire
        potential_contribution = min(agent_fire_reduction_power * agent_suppressant_num, f_intensity)
        
        # Scoring formula: considers distance, fire putout weight, and potential contribution
        # A small epsilon is added to distance to avoid division by zero
        score = (f_weight * potential_contribution) / (distance + 0.1)
        
        # Update the best task if this task's score is higher than the current best score
        if score > best_score:
            best_score = score
            best_task = i
            
    # In rare cases where no suitable task is found, default to the first task
    if best_task is None:
        best_task = 0

    return best_task