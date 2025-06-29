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
    
    import numpy as np
    
    num_tasks = len(fire_pos)
    best_task_index = -1
    best_task_score = -float('inf')
    
    proximity_temp = 1
    intensity_temp = 0.1
    resource_temp = 1
    weight_temp = 1
    
    for idx in range(num_tasks):
        # Calculate distance from agent to the fire
        distance = np.sqrt(np.sum(np.power(np.array(fire_pos[idx]) - np.array(agent_pos), 2)))
        proximity_score = 1 / (1 + distance)
        
        # Normalized intensity score: lower remaining intensity after effort
        possible_reduction = agent_suppressant_num * agent_fire_reduction_power
        expected_remaining_intensity = fire_levels[idx] - possible_reduction
        intensity_score = 1 / (1 + np.exp(intensity_temp * (expected_remaining_intensity - fire_levels[idx])))
        
        # Agent's resource utilization score
        resource_score = 1 / (1 + np.exp(resource_temp * (agent_suppressant_num - possible_reduction)))
        
        # Task priority/wieight score
        weight_score = fire_putout_weight[idx]
        
        # Combining the scores
        task_total_score = (
            proximity_score * np.exp(-proximity_temp * distance) + 
            intensity_score + resource_score + weight_score * weight_temp
        )
        
        # Choosing the task with the maximum score
        if task_total_score > best_task_score:
            best_task_score = task_total_score
            best_task_index = idx
    
    return best_task_index