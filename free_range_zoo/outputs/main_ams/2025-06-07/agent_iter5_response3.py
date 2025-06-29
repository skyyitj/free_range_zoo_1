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
    normalized_distances = 1 / (np.array(distances) + 0.001)  # Add a small value to avoid division by zero

    # Calculate potential suppressant effect on each fire
    potential_fire_reductions = agent_suppressant_num * agent_fire_reduction_power / np.array(fire_intensities)
    
    # Compute how effective suppressant can reduce the fire intensity
    new_fire_levels = np.array(fire_levels) - potential_fire_reductions

    # Normalize the suppression effect to include in score
    # Getting rid of clipping, normalizing by maximum potential reduction
    suppressant_effectiveness = potential_fire_reductions / np.max(fire_levels)

    # Calculate how desperate a fire needs to be addressed based on current intensity
    fire_desperation = np.array(fire_levels) / np.max(fire_levels)

    # Score function revision
    # Increased weight on desperation (multiplied by 2.0) to address more intense fires.
    # Normalized distances increase the value of closer fires.
    scores = normalized_distances * (np.array(fire_putout_weight) * suppressant_effectiveness * 2.0 + fire_desperation)

    # Select the fire task with the highest score
    # Modification: Adding an exploration factor improved by adding a random fluctuation to each score
    exploration_factor = 0.05  # 5% random margin
    random_noise = np.random.rand(len(scores)) * exploration_factor
    scores += random_noise

    selected_task_index = np.argmax(scores)

    return selected_task_index