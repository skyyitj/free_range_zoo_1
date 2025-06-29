def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_supressant_num: float, 
    other_agents_pos: List[Tuple[float, float]], 
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')
    
    # Adjust temperature tuning based on feedback
    distance_temp = 1.0  # Less sensitivity to distance to directly prioritize closer fires
    effectiveness_temp = 0.5  # Decrease to encourage better usage of large suppressant amounts
    importance_temp = 2.0    # Increase importance to factor in priority weights significantly

    # Strategy was to use putting out high-weight fires in priority, during resource allocation
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate the Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        # Estimate potential suppressant use, depending on fire intensity and what's left
        feasible_suppressant_use = min(agent_supressant_num, fire_intensity / agent_fire_reduction_power)
        if feasible_suppressant_use < 1:
            continue  # Skip if feasible suppressant is too low to make a difference

        potential_effectiveness = agent_fire_reduction_power * feasible_supressant_use
        importance_weight = fire_putout_weight[task_index]
        
        # Calculate task score based on various weights (emphasizing rewards and effectiveness)
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(potential_effectiveness + 1) * effectiveness_temp +
            importance_weight * 5.0 * importance_temp  # Great emphasis on importance weights
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index