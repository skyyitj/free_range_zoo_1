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

    # Scale factor and temperature value for Distance normalization to enhance closer fire preference
    distance_scale_factor = 1.5
    normalized_distances = np.exp(-np.array(distances) * distance_scale_factor)

    # Assess suppressant sufficiency for each fire
    suppressant_needs = np.array(fire_intensities) / agent_fire_reduction_power
    suppressant_sufficiency = np.where(agent_supressant_num >= suppressant_needs, 1, agent_supressant_num / suppressant_needs)

    # Calculate the impact scores using fire levels, suppressant sufficiency, and normalized distances
    scores = normalized_distances * suppressant_sufficiency * np.array(fire_putout_weight)

    # Select fire with maximum score
    selected_task_index = np.argmax(scores)

    return selected_task_index