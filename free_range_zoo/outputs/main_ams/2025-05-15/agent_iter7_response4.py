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

    # Tuning factors
    distance_temperature = 0.02
    intensity_temperature = 0.02
    suppressant_conserve_factor = 50  # Encouraging conservation
    reward_scale = 3  # Intensifies focus on reward weights

    # Compensating for available suppressants relative to needs
    suppressant_potential = (agent_suppressant_num / agent_fire_reduction_power) if agent_fire_reduction_power != 0 else 0

    for i in range(num_tasks):
        # Calculate Euclidean distance from agent to fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        # Adjust intensity use and impact
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(intensity_temperature * intensity)  # Optimistic (expands high values)

        # Calculate task score with tuned parameters
        score = reward_scale * fire_putout_weight[i] * norm_distance * norm_intensity * suppressant_potential

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index