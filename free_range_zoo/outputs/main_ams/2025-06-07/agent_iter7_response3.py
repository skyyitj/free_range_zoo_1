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

    # Calculate distances of the agent to each fire
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances to prioritize nearby fires
    distance_temp = 0.1  # Lower temperature makes distant penalties less severe
    normalized_distances = 1 / (1 + np.exp((np.array(distances) - np.mean(distances)) / distance_temp))

    # Calculate potential suppressant effect on each fire
    suppressant_effect = agent_suppressant_num * agent_fire_reduction_power
    normalized_fire_intensities = fire_intensities / np.max(fire_intensities)  # Normalize fire difficulties
    
    # Suppressant needed to handle fires based on their intensity
    suppressant_needed = np.array(fire_intensities) / agent_fire_reduction_power
    suppressant_enough = suppressant_effect >= suppressant_needed

    # Normalize fire levels forecasted after agent intervention
    remaining_intensities = np.maximum(np.array(fire_levels) - suppressant_effect, 0)
    normalized_remaining_intensities = remaining_intensities / np.max(fire_levels if np.max(fire_levels) else 1)

    # Calculate the effectiveness weight on remaining intensities and needs
    effectiveness_weight = suppressant_enough.astype(float) * (1 - normalized_remaining_intensities)
    effectiveness_temp = 0.5  # medium sensitivity to effectiveness
    effectiveness_scores = np.exp(effectiveness_weight / effectiveness_temp)

    # Combine all scores to determine which fire to target
    # A higher weight means a higher priority, effectiveness should also significantly contribute
    reward_ratio = np.array(fire_putout_weight) / np.max(fire_putout_weight)  # normalize rewards to max weight
    scores = reward_ratio * effectiveness_scores * normalized_distances
    
    # Selection of the fire task with the highest score
    selected_task_index = np.argmax(scores)

    return selected_task_index