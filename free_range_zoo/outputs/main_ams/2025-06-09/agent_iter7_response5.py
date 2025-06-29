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

    # Re-adjusting the temperature constants
    distance_temp = 0.3  # Further increase importance of close fires
    reduction_power_temp = 2.0  # New temperature tuning for reduction effectiveness
    suppressant_eff_temp = 2.5  # Further emphasis on efficient use of suppressants
    importance_temp = 4.0  # Enhanced importance scaling

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_supressant_num)
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        suppressant_efficiency = potential_effectiveness / target_suppressant_use if target_suppressant_use > 0 else 0
        importance_weight = fire_putout_weight[task_index]
        
        # Re-evaluate the score with new temp settings
        task_score = (
            -np.exp(distance) / distance_temp +
            np.log(potential_effectiveness + 1) * reduction_power_temp +
            np.log(suppressant_efficiency + 1) * suppressant_eff_temp +
            np.exp(importance_weight) * importance_temp
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index