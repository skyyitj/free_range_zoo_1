from typing import Tuple, List

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
    # Distance temperature normalization factor
    distance_temperature = 0.5
    # Intensity temperature normalization factor
    intensity_temperature = 5.0
    # Initialize best score and task index
    best_score, task_index = float('-inf'), -1
    
    # Iterate over each fire task to compute scores
    for i, (fire_y, fire_x) in enumerate(fire_pos):
        # Calculate the distance to the fire
        distance = ((agent_pos[0]-fire_y)**2 + (agent_pos[1]-fire_x)**2)**0.5
        # Normalize distance and apply exponential decay to prioritize closer fires
        distance_score = 1.0 / (1.0 + distance ** distance_temperature)
        
        # Evaluate the agent's capability against the fire intensity
        capability_score = agent_fire_reduction_power / (1.0 + fire_intensities[i] ** intensity_temperature)
        
        # Assess resource availability
        resource_score = agent_suppressant_num / (1.0 + fire_levels[i] ** intensity_temperature)
        
        # Compute weighted priority of the fire
        priority_score = fire_putout_weight[i]
        
        # Calculate combined score for the task, valuing each aspect
        score = distance_score * capability_score * resource_score * priority_score
        
        # Check if the current score is the best and update task indices accordingly
        if score > best_score:
            best_score = score
            task_index = i
            
    return task_index