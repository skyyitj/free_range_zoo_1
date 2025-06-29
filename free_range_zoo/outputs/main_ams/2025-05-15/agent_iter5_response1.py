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

    # Adjusted temperature parameters
    distance_temperature = 0.001  # Reduced to prioritize distant high-weight fires
    intensity_temperature = 0.01  # Slightly reduce impact of intensity
    reward_scale = 10  # Scaling up rewards significance in decisions
    suppressant_factor = 10.0  # Encourage conserving suppressants slightly less aggressively

    normalized_suppressant_usage = np.exp(-suppressant_factor * (1 - agent_suppressant_num / 10))
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)
        
        # Complex function to evaluate task attractiveness based on fire intensity, rewards, and resources
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)
        
        # Include reward_weight more significantly into consideration
        score = (fire_putout_weight[i] * reward_scale) * (norm_distance * agent_fire_reduction_power / (1 + intensity)) * np.log1p(fire_levels[i]) * normalized_suppressant_usage * norm_intensity
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index