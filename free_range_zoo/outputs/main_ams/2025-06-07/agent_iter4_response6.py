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

    # Calculate distances of the agent to each fire
    distances = np.array([
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ])

    # Inverse distance weighting: prioritize nearer fires
    distance_score_scale = 10.0
    distance_scores = 1 / (1 + distances * distance_score_scale)

    # Estimate impact of agent's suppressant on the fire
    potential_fire_intensity = np.array(fire_intensities) - (agent_supressant_num * agent_fire_reduction_power)

    # Fire suppressant effectiveness scores: prioritize fires that can most likely be decreased or extinguished
    suppressant_effectiveness_scores = np.maximum(0, np.sign(potential_fire_intensity))

    # Choiry weight contribution
    weighted_contributions = fire_putout_weight * suppressant_effectiveness_scores

    # Combine all scores: distance factor ensures proximity has priority
    combined_scores = distance_scores * weighted_contributions

    # Normalize combined scores to range [0, 1] to ensure no overflow in decision making
    if np.max(combined_scores) > 0:
        combined_scores /= np.max(combined_scores)

    # Use a softmax scaling approach to make a probabilistic decision, be biased toward the highest values
    score_temperature = 0.5  # low temperature sharpens the choice around high values
    probabilities = np.exp(combined_scores / score_temperature) / np.sum(np.exp(combined_scores / score_temperature))

    # Choose according to probabilities
    selected_task_index = np.argmax(probabilities)

    return selected_task_index