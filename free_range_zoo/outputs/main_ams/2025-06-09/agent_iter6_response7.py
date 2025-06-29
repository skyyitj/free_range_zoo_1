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

    # Adjusting the temperature constants optimally
    distance_temp = 0.5  # More sensitive to distance to prioritize nearby tasks
    effectiveness_temp = 3.0  # Increase the influence of potential effectiveness
    importance_temp = 3.0  # More emphasis on the prioritization of tasks based on their associated rewards
    suppressant_eff_temp = 2.5  # Increase the weight of suppressant efficiency which is crucial

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Calculate the maximum suppressant the agent can use effectively
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_suppressant_num)
        
        # Effective reduction in fire intensity which can be achieved
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        # Calculate suppressant efficiency
        if target_suppressant_use > 0:
            suppressant_efficiency = potential_effectiveness / target_suppressant_use
        else:
            suppressant_efficiency = 0 
        
        importance_weight = fire_putout_weight[task_index]
        
        # Comprehensive task score calculation
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(potential_effectiveness + 1) * effectiveness_temp +
            np.log(suppressant_efficiency + 1) * suppressant_eff_temp +
            importance_weight * importance_temp
        )
        
        # Selecting the best task index based on computed score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index