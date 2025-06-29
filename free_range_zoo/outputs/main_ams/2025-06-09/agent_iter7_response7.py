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

    # Adjusted temperature constants for better control and balance between metrics
    distance_temp = 0.3  # Increase sensitivity to distance further
    effectiveness_temp = 5.0  # Higher emphasis on having an immediate impact on fires
    importance_temp = 4.0  # Increase weighting of higher priority fires
    suppressant_eff_temp = 2.5  # Greater focus on efficient resource usage

    # Loop over each fire to calculate the potentially best task for the agent
    for task_index in range(num_tasks):
        fire_location = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance from agent to each fire 
        distance = np.sqrt((agent_pos[0] - fire_location[0])**2 + (agent_pos[1] - fire_location[1])**2)

        # Calculate how much intensity can be reduced effectively with available suppressants
        effective_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_suppressant_num)
        potential_reduction = agent_fire_reduction_power * effective_suppressant_use
        
        if effective_suppressant_use > 0:
            suppressant_efficiency = potential_reduction / effective_suppressant_use
        else:
            suppressant_efficiency = 0 

        importance_weight = fire_putout_weight[task_index]
        
        # Formulate the score considering all the factors with their temperature adjustments
        task_score = (
            -np.exp(distance / distance_temp) +  # Increase sensitivity to shorter distances
            np.log(potential_reduction + 1) * effectiveness_temp +  # Emphasizing on maximal fire reduction
            np.exp(suppressant_efficiency) * suppressant_eff_temp +  # Rewarding efficient use of fire suppressants
            np.exp(importance_weight) * importance_temp  # Heavier weighting for higher priority tasks
        )
        
        # Choose the fire task with the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index