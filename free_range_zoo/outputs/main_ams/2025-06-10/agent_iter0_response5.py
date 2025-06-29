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

    num_tasks = len(fire_pos)
    
    # Constants and parameters used for scoring tasks
    distance_scale = 1.0  # Scaling factor for distance penalty
    intensity_scale = 1.0  # Scaling factor for fire intensity importance
    remaining_suppressant_scale = 1.2  # Importance of preserving suppressant
    level_weight_scale = 1.0  # Scaling for fire level acceptance
    weight_temp = 0.5  # Temperature for exponential weighting of fire putout weights
    
    # List of scores for each fire task
    scores = []
    
    for i in range(num_tasks):
        # Calculate distance to the fire spot from the agent's current position
        fire_distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        
        # Consider the remaining amount of suppressant post engagement with this fire
        potential_suppressant_remaining = max(0, agent_suppressant_num - fire_intensities[i] / agent_fire_reduction_power)
        
        # Compute scorings
        distance_penalty = np.exp(-distance_scale * fire_distance)
        intensity_importance = np.exp(-intensity_scale * fire_intensities[i])
        suppressant_preservation = np.exp(remaining_suppressant_scale * potential_suppressant_remaining)
        level_weight_importance = np.exp(-level_weight_scale * fire_levels[i])
        
        # Weighted importance of putting out this fire (considering policy weights for tasks)
        weighted_importance = np.exp(weight_temp * fire_putout_weight[i])
        
        # Composite score for task selection
        score = distance_penalty * intensity_importance * suppressant_preservation * level_weight_importance * weighted_importance
        
        scores.append(score)
    
    # Select the task with the highest score
    best_task_index = int(np.argmax(scores))
    
    return best_task_index