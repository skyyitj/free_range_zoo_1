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

    distance_temp = 0.8
    importance_temp = 3.0
    effectiveness_temp = 2.5
    efficiency_temp = 2.0

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate distance and derived score component
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        inverse_distance_score = np.exp(-distance / distance_temp)
        
        # Efficient use of suppressant to maximize impact
        effective_suppressant_use = min(fire_intensity / agent_fire_reduction_power, agent_supressant_num)
        effective_fire_reduction = effective_suppressant_use * agent_fire_reduction_power
        efficiency_score = np.log(effective_fire_reduction + 1) * effectiveness_temp
        
        importance_score = (fire_putout_weight[task_index] + 1) * importance_temp

        # Calculate a compound score
        task_score = (inverse_distance_score + efficiency_score + importance_score)
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index