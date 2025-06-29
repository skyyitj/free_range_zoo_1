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

    # Improved policy uses distance, suppressant efficiency, fire levels, and reward weights more effectively.
    
    # Calculate distances of the agent to each fire
    distances = np.array([
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ])
    
    # Normalize distances: closer fires have higher priority, avoid division by zero by adding a small value
    normalized_distances = 1 / (distances + 0.1)

    # Calculate potential fire level reduction with the available suppressant
    possible_reduction = agent_suppressant_num * agent_fire_reduction_power

    # Evaluate the suppressant effect on each fire; proportional reduction considering fire intensity
    suppressant_effectiveness = possible_reduction / np.array(fire_intensities)
    suppressant_effectiveness[suppressant_effectiveness > 1] = 1  # Cap the effectiveness to maximum extinguishability

    # Calculate the remaining fire levels after suppression attempts
    remaining_fire_levels = np.array(fire_levels) - suppressant_effectiveness * np.array(fire_levels)
    
    # Fires that are expected to be addressed reliably are weighted higher
    extinguishable_scores = 1 / (1 + np.exp(10 * (remaining_fire_levels - 0.1)))

    # Combine normalized distances, suppression effectiveness, and rewards, adjusted by task weights
    scores = fire_putout_weight * normalized_distances * extinguishable_scores

    # Select the fire task with the highest score
    selected_task_index = np.argmax(scores)
    
    return selected_task_index