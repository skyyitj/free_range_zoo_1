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

    # Adjust leaders for balancing between effectiveness, efficiency, and strategic positioning
    distance_temp = 0.5  # Lower influence of distance
    effectiveness_temp = 3.0  # Increase the importance of effectiveness
    importance_temp = 3.0  # Increase weights assigned to importance
    suppressant_eff_temp = 2.0  # Increase focus on suppressant efficiency

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance to the fire
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Calculate maximum effective suppressant usage
        max_effective_suppressant = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_supressant_num)
        
        # Effective potential reduction in fire intensity
        potential_effectiveness = agent_fire_reduction_power * max_effective_suppressant
        
        # Calculating suppressant efficiency
        if max_effective_suppressant > 0:
            suppressant_efficiency = potential_effectiveness / max_effective_suppressant
        else:
            suppressant_efficiency = 0 
        
        importance_weight = fire_putout_weight[task_index]
        
        # Constructing the task score
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(potential_effectiveness + 1) * effectiveness_temp +
            np.log(suppressant_efficiency + 1) * suppressant_eff_temp +
            importance_weight * importance_temp
        )
        
        # Decide on the best task based on the calculated score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index