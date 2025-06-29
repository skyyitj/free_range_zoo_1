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

    # Compute distances from the agent to each fire
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances to convert into scores (smaller distances should have higher scores)
    normalized_distances = 1 / (np.array(distances) + 0.01)  # Avoid division by zero

    # Estimate the effectiveness of the agent on each fire, considering remaining suppressant
    effectivity_scores = np.zeros(len(fire_levels))
    for i, fire_intensity in enumerate(fire_intensities):
        potential_reduction = agent_supressant_num * agent_fire_reduction_power / fire_intensity
        remaining_fire_intensity = fire_levels[i] - potential_reduction
        if remaining_fire_intensity <= 0:
            effectivity_scores[i] = 5  # High score for fully manageable fires
        else:
            # Scaled down score for non-extinguishable fires
            effectivity_scores[i] = 1 / (remaining_fire_intensity + 1)

    # Combine scores incorporating priority weights and normalized distances
    combined_scores = normalized_distances * fire_putout_weight * effectivity_scores

    # Choose the fire index with the maximum combined score
    selected_fire_index = np.argmax(combined_scores)

    return selected_fire_index