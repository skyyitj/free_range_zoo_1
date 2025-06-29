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

    # Finely-tuned temperature parameters
    distance_temperature = 0.005  # refined for balance in task proximity preference
    intensity_temperature = 0.01  # reduced to focus more on severe fires
    suppressant_conserve_factor = 8.0  # modified for better resource management and efficiency

    # Calculate a normalized suppressant conservation metric
    remaining_suppressant = np.exp(-suppressant_conserve_factor * (1 - (agent_suppressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        # Using log1p for smoother distance effect and avoiding heavy penalties on closer tasks
        norm_distance = np.exp(-distance_temperature * np.log1p(distance))

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)
        
        # Factor in reward potential with intensity and suppressant considerations
        score = (fire_putout_weight[i] * norm_distance * np.log1p(agent_fire_reduction_power / (1 + intensity)) *
                 norm_intensity * remaining_suppressant)
  
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index