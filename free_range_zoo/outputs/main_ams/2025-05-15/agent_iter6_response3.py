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

    # Enhanced parameters
    distance_temperature = 0.01  # smaller to emphasize closer fires
    intensity_temperature = 0.01  # smaller to emphasize higher intensity fires more prominently
    suppressant_conserve_factor = 40.0  # Higher to conserve suppressant significantly
    reward_scale = 2.0  # Enhanced focus on higher rewards

    # Compute a suppression potential metric
    suppressant_potential = np.exp(-suppressant_conserve_factor * (1 - (agent_suppressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        intensity = fire_intensities[i] * fire_levels[i]  # Adjusted to consider both prop values
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # Calculate score considering adjusted factors
        score = reward_scale * fire_putout_weight[i] * (norm_distance + norm_intensity) * suppressant_potential

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index