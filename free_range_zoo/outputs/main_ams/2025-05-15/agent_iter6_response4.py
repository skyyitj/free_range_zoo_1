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

    # Tuned parameters
    distance_temperature = 0.002  # More receptive to distant fires if they score high elsewhere
    intensity_focus_scaling = 100  # Sharpen focus on high-intensity fires
    reward_weight_scale = 1.8  # Increase the influence of reward weights in decision-making
    suppressant_preservation_factor = 30.0  # More focus on conserving suppressant

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_focus_scaling / intensity)

        expected_suppressant_usage = (1 + intensity) / agent_fire_reduction_power
        suppressant_efficiency = agent_suppressant_num / expected_suppressant_usage
        
        score = (reward_weight_scale * fire_putout_weight[i] * norm_distance *
                 suppressant_efficiency * norm_intensity)

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index