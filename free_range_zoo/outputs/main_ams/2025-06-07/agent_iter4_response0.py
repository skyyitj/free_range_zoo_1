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

    # Normalize the suppression effect; fires expected to go out are highly valuable
    suppressant_effectiveness = np.clip(new_fire_levels, 0, None) == 0

    # Calculate how desperate a fire needs to be addressed based on current intensity
    fire_desperation = np.array(fire_levels) / np.max(fire_levels)

    # Combine all scores to determine which fire to target
    # Incorporate fire importance, suppressant effectiveness directly, and desperation inversely
    scores = normalized_distances * (np.array(fire_putout_weight) * suppressant_effectiveness + fire_desperation)
    
    # Selection of the fire task with the highest score
    selected_task_index = np.argmax(scores)

    return selected_task_index