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

    # Dynamic parameter adjustments based on runtime evaluation
    distance_temp = 0.02
    intensity_temp = 0.03

    for i in range(num_tasks):
        fire_distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        normalized_distance = np.exp(-distance_temp * fire_distance)

        fire_total_intensity = fire_intensities[i] * fire_levels[i]
        normalized_intensity = np.exp(-intensity_temp * fire_total_intensity)

        suppressant_factor = agent_suppressant_num / max(fire_total_intensity, 1)  # Prevent division by zero

        score = (fire_putout_weight[i] * normalized_intensity * normalized_distance *
                 agent_fire_reduction_power * np.log1p(suppressant_factor))
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index