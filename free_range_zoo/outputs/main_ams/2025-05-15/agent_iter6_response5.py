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

    # Adjusting parameters to better modulate decision factors
    distance_temperature = 0.005  # Maintains precision in distance considerations
    intensity_temperature = 0.1   # More responsive to varying intensities
    resource_conservation_weight = 50.0  # Stronger focus on suppressant conservation
    reward_imperative = 2.0       # Enhanced focus on better rewarding tasks

    remaining_suppressant = np.exp(-resource_conservation_weight * (1 - (agent_suppressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # Enhancing the significance of high-reward & critical tasks
        reward_focus = reward_imperative * fire_putout_weight[i]
        suppressant_factor = remaining_suppressant * norm_intensity
        
        score = reward_focus * norm_distance * agent_fire_reduction_power / (1 + intensity) * np.log1p(fire_levels[i]) * suppressant_factor

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index