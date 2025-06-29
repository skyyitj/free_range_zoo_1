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

    # Tuning temperature scale parameters based on metric feedback
    distance_temp = 0.5  # Further reduce effect of distance
    effectiveness_temp = 2.0  # Maintained effectiveness importance
    importance_temp = 3.0  # Increase importance of priority (rewards)
    suppressant_eff_temp = 2.5  # Increase focus on suppressant efficiency

    for task_index in range(num_tasks):
        # Extract each fire's stats
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]

        # Calculate Euclidean distance to the fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Calculate the ideal suppressant to use based on required reduction and capacity
        efficient_use_of_suppressant = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_supressant_num)

        # Estimate fire suppression potential
        effective_fire_reduction = agent_fire_reduction_power * efficient_use_of_suppressant
        
        # Calculate suppressant efficiency
        if efficient_use_of_suppressant > 0:
            suppressant_efficacy_rate = effective_fire_reduction / efficient_use_of_suppressant
        else:
            suppressant_efficacy_rate = 0 
        
        # Extract the importance weight associated with the fire
        task_importance = fire_putout_weight[task_index]
        
        # Score for selecting the task
        score = (
            -np.log(distance + 1) / distance_temp +
            np.log(effective_fire_reduction + 1) * effectiveness_temp +
            np.log(suppressant_efficacy_rate + 1) * suppressant_eff_temp +
            np.log(task_importance + 1) * importance_temp
        )
        
        # Choosing the maximum score fire task
        if score > highest_score:
            highest_score = score
            best_task_index = task_index

    return best_task_index