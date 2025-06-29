def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_supressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Adjusted parameters for better policy efficiency
    distance_scale = 0.05  # focus more on distant but important fires
    intensity_focus = 1.0  # stark focus on intensity
    reward_scale = 3.0  # triple emphasis on reward weight
    
    # Utilize a dynamic assessment for suppressant use
    suppressant_use_factor = max(1.0, agent_fire_reduction_power / 5.0 * (agent_suppressant_num / 5))

    for i in range(num_tasks):
        # Calculate effective distance impacting factor
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        distance_factor = np.log1p(distance) * distance_scale

        # Calculate reward influenced intensity factor
        intensity = fire_intensities[i] * fire_levels[i]
        effective_intensity = np.log1p(intensity) * intensity_focus

        # Compute task score evaluating distance, intensity, and resource management
        score = reward_scale * fire_putout_weight[i] / (distance_factor + effective_intensity + suppressant_use_factor)
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index