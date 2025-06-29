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
    
    # Determine number of fires
    num_fires = len(fire_pos)
    
    # Calculate distances to all fires
    distances = np.array([
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ])
    
    # Normalize distances
    distance_weights = np.exp(-distances / 10.0)  # Using a new temperature constant of 10.0
    
    # Evaluate efficiency of suppression per suppressant used
    suppression_scores = np.array([
        min(agent_fire_reduction_power * agent_suppressant_num, intensity)
        for intensity in fire_intensities
    ])
    
    # Calculate potential fire extinction effectiveness
    effectiveness = suppression_scores / np.maximum(fire_intensities, 1e-6)
    
    # Weigh scores by priority weights given to put out fires
    priority_adjusted_scores = effectiveness * np.array(fire_putout_weight)
    
    # Combine priority scores with distance weights
    final_scores = priority_adjusted_scores * distance_weights
    
    # Handle suppressant conservation by amplifying focus when resources are scarce
    suppressant_scaler = (agent_suppressant_num + 1) / (np.mean([agent_suppressant_num for _ in range(num_fires)]) + 1)
    final_scores *= suppressant_scaler 
    
    # Identify the highest scoring fire to fight
    chosen_fire_index = np.argmax(final_scores)

    return chosen_fire_index