def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    import numpy as np

    num_fires = len(fire_pos)
    task_scores = np.zeros(num_fires)
    
    # Get the distances from the agent to each fire
    agent_y, agent_x = agent_pos
    distances = np.array([np.sqrt((agent_y - fy) ** 2 + (agent_x - fx) ** 2) for fy, fx in fire_pos])
    
    # Constants and scales for adjustments
    intensity_scale_temp = 0.05  # Temperature constant for scaling 'fire_intensities'
    distance_scale_temp = 1.0    # Temperature constant for scaling 'distances'
    suppressant_scale_temp = 0.1 # Temperature constant for scaling agent suppression resource
    level_scale_temp = 0.25      # Temperature for scaling fire levels
    
    for i in range(num_fires):
        fire_intensity = fire_intensities[i]
        fire_distance = distances[i]
        fire_level = fire_levels[i]
        weight = fire_putout_weight[i]

        # Calculating scaled parts (transformations)
        scaled_intensity = np.exp(-fire_intensity * intensity_scale_temp)
        scaled_distance = np.exp(-fire_distance * distance_scale_temp)
        scaled_level = np.exp(fire_level * level_scale_temp)

        # Compound score calculation for this fire, considering distance, intensity, and suppression weight
        task_scores[i] = weight * scaled_intensity * scaled_distance * scaled_level
    
    # The preference is the fire with the highest score
    selected_task_index = int(np.argmax(task_scores))
    
    return selected_task_index