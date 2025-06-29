def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_supressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    import numpy as np

    # Calculate distances between the agent and each fire location
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances so closer fires are more attractive
    # A higher weight to nearby fires to improve efficiency
    max_distance = np.max(distances) if np.max(distances) else 1  # Avoid division by zero
    normalized_distances = (1 - (np.array(distances) / max_distance)) ** 2

    # Calculate the potential to control each fire, adjusting for the agent's limited resources
    estimated_impact = np.array(fire_levels) - agent_fire_reduction_power * min(agent_supressant_num, np.array(fire_intensities))
    estimated_impact = np.clip(estimated_impact, 0, None)  # Keep at least at zero

    # Normalize the estimated impact
    required_impact = np.clip(fire_levels - estimated_impact, 0, None)
    max_required_impact = np.max(required_impact) if np.max(required_impact) else 1
    normalized_required_impact = required_impact / max_required_impact
    
    # Incorporate reward weights with a focus on significant weight adjustments
    # Reward weights are adjusted by exponential scale to emphasize more important tasks
    exp_weights = np.exp(np.array(fire_putout_weight))
    normalized_weights = exp_weights / np.sum(exp_weights)

    # Calculate combined scores favoring higher weights, closer distances, and feasible fire suppression
    scores = normalized_distances * normalized_required_impact * normalized_weights

    # Applying temperature scale to further refine the decision sensitivity
    temperature = 0.5  # A smaller temperature can sharpen the preference
    exp_scores = np.exp(scores / temperature)
    probabilities = exp_scores / np.sum(exp_scores)

    # Selecting the fire with the maximum probability after temperature scaling
    selected_task_index = np.argmax(probabilities)

    return selected_task_index