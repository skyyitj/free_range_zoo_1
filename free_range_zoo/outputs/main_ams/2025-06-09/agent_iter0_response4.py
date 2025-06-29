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

    import numpy as np

    num_tasks = len(fire_pos)
    if num_tasks == 0:
        return -1  # No tasks available
    
    best_task_index = -1
    highest_score = float('-inf')
    
    score_temp = 0.5
    
    for task_index in range(num_tasks):
        fire_y, fire_x = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        fire_level = fire_levels[task_index]
        weight = fire_putout_weight[task_index]
        
        dx = fire_x - agent_pos[1]
        dy = fire_y - agent_pos[0]
        distance = np.sqrt(dx**2 + dy**2)
        
        if fire_intensity <= 0:
            continue  # Skip extinguished fires
        
        suppressant_needed = fire_intensity / agent_fire_reduction_power
        if suppressant_needed > agent_suppressant_num:
            effectiveness = agent_suppressant_num * agent_fire_reduction_power / fire_intensity
        else:
            effectiveness = 1  # Can fully handle the fire

        # Score calculation: consider distance, effectiveness, weight of the fire, and fire level
        score = weight * effectiveness / (1 + distance)
        
        # Optionally adjusting score by applying a non-linear transformation for better prioritization
        normalized_score = np.exp(score_temp * score)
        
        if normalized_score > highest_score:
            highest_score = normalized_score
            best_task_index = task_index
            
    return best_task_index