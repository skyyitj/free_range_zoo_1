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

    # Recalibrated temperature constants for better control
    distance_temp = 0.1     # Increase influence of distance to prefer closer fires
    effectiveness_temp = 5.0 # Increasing the importance of potential effectiveness
    importance_temp = 4.0    # Put more emphasis on specified importance weights
    
    suppressant_eff_temp = 2.0  # Adjustment in suppressant efficiency scale

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Evaluate the maximum effective suppressant use
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_suppressant_num)
        
        # Calculate potential effectiveness
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        # Calculate suppressant efficiency
        if target_suppressant_use > 0:
            suppressant_efficiency = potential_effectiveness / target_suppressant_use
        else:
            suppressant_efficiency = 0 
        
        importance_weight = fire_putout_weight[task_index]
        
        # Score calculation incorporating suppressant efficiency
        task_score = (
            -np.exp(-distance / distance_temp) +
            np.exp(potential_effectiveness * effectiveness_temp) +
            np.log1p(suppressant_efficiency) * suppressant_eff_temp +
            np.exp(importance_weight * importance_temp)
        )
        
        # Select the task with the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index