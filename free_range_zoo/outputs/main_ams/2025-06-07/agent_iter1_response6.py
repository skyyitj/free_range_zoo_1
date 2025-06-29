def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np
    
    # Configure optimal parameters that can be tuned for better performance
    distance_weighting_temp = 5.0
    suppressant_use_temp = 8.0  # Reduced for more agile suppressant use response
    
    # Calculate distances from agent to each fire
    distances = [np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2) for f_pos in fire_pos]

    # Compute scores based on weight, suppressant potential, and fire level
    scores = []
    max_intensity = max(fire_intensities) if fire_intensities else 1  # Ensure no division by zero
    
    for idx in range(len(fire_pos)):
        distance = distances[idx]
        fire_level = fire_levels[idx]
        intensity = fire_intensities[idx]
        weight = fire_putout_weight[idx]
        
        # use exponential decay for distance impact
        distance_factor = np.exp(-distance / distance_weighting_temp)
        
        # Suppression potential is the estimated reduction from our agent
        suppression_potential = min(intensity, agent_fire_reduction_power * agent_suppressant_num)
        suppression_factor = suppression_potential / max_intensity
        
        # Include the potential for putting out a fire completely
        complete_extinguish_bonus = 5.0 if suppression_potential >= intensity else 1.0
        
        # Compute final score influenced by task weight and fire importance
        score = (weight * suppression_factor * distance_factor) * complete_extinguish_bonus
        
        # Consider suppressant use efficiency
        suppressant_use_factor = np.exp(-(agent_fire_reduction_power / (agent_suppressant_num + 1e-6)) / suppressant_use_temp)
        effective_score = score * suppressant_use_factor
        
        scores.append(effective_score)

    # Select the highest score index
    selected_task_index = np.argmax(scores)
    return selected_task_index