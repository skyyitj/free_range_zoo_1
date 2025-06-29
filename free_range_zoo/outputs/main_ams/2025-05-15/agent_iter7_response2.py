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

    # Adjust parameters based on observed metrics
    distance_weight = 0.08 
    intensity_boost_factor = 3.0  # Increased focus on intensity
    conservation_factor = 25.0    # Adjusted suppressant conservation behavior
    weight_scale = 3.0            # Adjust to leverage available reward info better

    # Calculate suppressant potential based on current levels
    suppressant_factor = np.exp(-conservation_factor * (1 - (agent_supressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        scaled_distance_score = np.exp(-distance_weight * distance)

        # Intensity scoring factoring level and base intensity
        intensity_score = fire_levels[i] * (fire_intensities[i] ** intensity_boost_factor)
        fire_priority_score = fire_putout_weight[i] * scaled_distance_score * intensity_score * suppressant_factor

        if fire_priority_score > best_task_score:
            best_task_score = fire_priority_score
            selected_task_index = i

    return selected_task_index