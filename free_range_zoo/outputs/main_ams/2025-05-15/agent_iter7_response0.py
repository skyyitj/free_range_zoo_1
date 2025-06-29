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
    best_task_score = float('-inf')
    selected_task_index = -1

    # Adjusted to focus more on high priority and high intensity/level fires
    distance_temperature = 0.01
    intensity_temperature = 0.02  # Increased importance on fire intensity and level
    suppressant_conserve_factor = 60.0  # More conservative suppressant use
    reward_scale = 5.0  # Increased influence of reward weight
    
    suppressant_potential = np.exp(-suppressant_conserve_factor * (1 - (agent_supressant_num / 100)))

    for i in range(num_tasks):
        distance = np.sqrt((fire_pos[i][0] - agent_pos[0])**2 + (fire_pos[i][1] - agent_pos[1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        # Increase the influence of fire severity
        combined_intensity_level = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * combined_intensity_level)
        
        # Calculating task score considering enhancements
        score = reward_scale * fire_putout_weight[i] * (norm_distance + norm_intensity) * suppressant_potential

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index