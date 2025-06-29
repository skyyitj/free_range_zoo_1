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
    # Calculate distances between the agent and each fire location
    import numpy as np
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances so closer fires are more attractive
    normalized_distances = np.exp(-np.array(distances) / 5.0)  # Temperature constant refined to better scale: 5.0
    
    # Boost scores by reduction potential and task weights
    suppression_potentials = np.minimum(np.array(fire_intensities), agent_fire_reduction_power * agent_supressant_num)
    normalized_suppression = suppression_potentials / np.max(fire_intensities) if np.max(fire_intensities) > 0 else suppression_potentials
    weighted_suppressions = normalized_suppression * np.array(fire_putout_weight)

    # Calculate final scores considering both distance, suppression potential, and usage of suppressants
    # Formula takes into account suppressant availability more efficiently
    consumed_suppressants = np.maximum(suppression_potentials / (agent_fire_reduction_power + 1e-6), 1e-6)
    suppressant_factor = 1 / (1 + (consumed_suppressants / (agent_supressant_num + 1e-6))) # More balanced accounting for suppressant usage
    scores = normalized_distances * weighted_suppressions * suppressant_factor

    # Selecting the fire with the maximum score
    selected_task_index = np.argmax(scores)

    return selected_task_index