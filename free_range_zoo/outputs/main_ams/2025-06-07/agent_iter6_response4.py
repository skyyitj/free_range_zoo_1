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

    # Compute distances from the agent to each fire
    distances = np.array([
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ])

    # Emphasize on closer fires using a reciprocal function (avoid zero division by adding a small constant)
    distance_factor = 1 / (distances + 0.1)

    # Normalize distance factor to further emphasize closer targets
    normalized_distance_factor = distance_factor / np.sum(distance_factor)

    # Efficiency of suppression based on agent's attributes and fire intensity
    suppression_efficiency = agent_fire_reduction_power / np.array(fire_intensities)

    # Anticipated reduction of fire level with given resources
    expected_suppression = suppression_efficiency * agent_suppressant_num

    # Calculate effective reduction as a proportion of existing fire level
    effective_fire_reduction = expected_suppression / np.array(fire_levels)

    # Modulate effective reduction with a logistic function to avoid trivializing higher values
    t_reduction = 1 / (1 + np.exp(-effective_fire_reduction))
    
    # Weights pulling agents towards more critical and rewarding fires
    weighted_importance = np.array(fire_putout_weight) * t_reduction

    # Final score combining all factors ensuring minimal suppressant usage
    scores = weighted_importance * normalized_distance_factor
    
    # Agent selects the fire with the highest score
    selected_task_index = np.argmax(scores)

    return selected_task_index