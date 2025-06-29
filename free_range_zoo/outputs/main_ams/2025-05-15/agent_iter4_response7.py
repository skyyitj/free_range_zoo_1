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

    # Parameter adjustments to address metrics:
    distance_temperature = 0.01  # Allow flexibility
    intensity_temperature = 0.05  # Focus slightly more on intense fires
    suppressant_conserve_factor = 10.0  # Encourage better suppressant usage

    # Normalize resources while considering their intensity reduction potential:
    remaining_suppressant = np.exp(-suppressant_conserve_factor * (agent_suppressant_num / max(fire_levels)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # modify the score calculation to prioritize put out larger fires and efficiently using suppressants
        score = (
            fire_putout_weight[i] * 
            (norm_distance * 1 / (1 + distance)) * 
            (norm_intensity * agent_fire_reduction_power / (1 + intensity)) * 
            remaining_suppressant
        )

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index