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
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')

    # Tuning these temperature constants based on evaluation metrics feedback
    distance_temp = 2.0  # Increase the influence of distance more to select closer fires
    effectiveness_temp = 1.0  # Reduce bias on this as it might lead to suppressant over-use
    importance_temp = 2.5  # Increase importance weighting to prioritize critical fires more
    
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate the Euclidean distance to the fire
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Consider the suppressant usage, aim for efficiency
        # Using a proportion of suppressant that is effective without overconsumption
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power * 0.8, agent_suppressant_num)
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        importance_weight = fire_putout_weight[task_index]
        
        # Score calculation integrates all enhanced parameters with respective temperatures
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(potential_effectiveness + 1) / effectiveness_temp +
            np.log(importance_weight + 1) * importance_temp
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index