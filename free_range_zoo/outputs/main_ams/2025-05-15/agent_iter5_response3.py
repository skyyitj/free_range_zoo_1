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
    selected_task_index = -1
    highest_priority_score = float('-inf')

    # Fine-tuning parameters
    distance_weight = 0.001   # Weaken the effect of distance slightly
    intensity_weight = 0.1    # Amplify the weight of fire intensity to prioritize critical fires
    suppressant_weight = 0.05  # Encourage more prudent suppressant use

    # Estimating remaining effective fire suppressant use (normalized)
    effective_suppressant = np.exp(-suppressant_weight * (agent_suppressant_num / 100))

    for i in range(num_tasks):
        # Distance calculation 
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        adjusted_distance = np.exp(-distance_weight * distance)
        
        # Adjusting task priority based on fire behavior and characteristics
        adjusted_fire_intensity = np.exp(intensity_weight * fire_intensities[i] * fire_levels[i])
        task_priority = fire_putout_weight[i] * adjusted_fire_intensity * adjusted_distance* effective_suppressant

        # Choose the task with the highest priority score
        if task_priority > highest_priority_score:
            highest_priority_score = task_priority
            selected_task_index = i

    return selected_task_index