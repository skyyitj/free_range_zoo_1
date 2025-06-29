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
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances inversely to prioritize nearby fires more strongly
    normalized_distances = 1 / (np.array(distances) + 0.01)  # Add a small value to avoid division by zero

    # Compute the reduction in fire levels considering the agent's suppression power and current suppressant
    effective_reduction = agent_supressant_num * agent_fire_reduction_power
    
    # The expected fire levels after agent's action
    expected_fire_levels = np.maximum(0.0, np.array(fire_levels) - effective_reduction / (np.array(fire_intensities) + 0.01))

    # Calculate utility of suppression based on the expected fire levels
    suppression_utility = (np.array(fire_levels) - expected_fire_levels) / (np.array(fire_intensities) + 0.01)

    # Normalize suppression utility
    max_utility = np.max(suppression_utility) if np.max(suppression_utility) > 0 else 1
    normalized_utility = suppression_utility / max_utility

    # Calculate scores incorporating distances, suppression potentials and designated task weights
    scores = normalized_distances * fire_putout_weight * normalized_utility

    # Selection of the fire task with the highest score
    selected_task_index = np.argmax(scores)

    return selected_task_index