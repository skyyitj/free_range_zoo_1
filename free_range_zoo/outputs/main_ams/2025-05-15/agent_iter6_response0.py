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
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Customized parameters to improve key metrics
    distance_temp = 0.001
    intensity_temp = 0.1  
    reward_scale = 2.5  
    suppressant_efficiency_factor = 15.0
    
    for i in range(num_tasks):
        fire_distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        normalized_distance = np.exp(-distance_temp * fire_distance)

        fire_intensity = fire_levels[i] * (fire_intensities[i] + 1)
        normalized_intensity = np.exp(intensity_temp * (1 / (fire_intensity + 1)))

        suppressant_efficiency = agent_suppressant_num / (fire_intensity + 1)
        normalized_suppressant_efficiency = np.exp(-suppressant_efficiency_factor * suppressant_efficiency)

        task_score = (reward_scale * fire_putout_weight[i] * normalized_distance *
                      agent_fire_reduction_power * normalized_suppressant_efficiency * normalized_intensity)

        if task_score > best_task_score:
            best_task_score = task_score
            selected_task_index = i

    return selected_task_index