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

    # Calculate distances between the agent and each fire location
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0])**2 + (f_pos[1] - agent_pos[1])**2)
        for f_pos in fire_pos
    ]

    # Normalize distances so closer fires are more attractive
    max_distance = np.max(distances) if np.max(distances) else 1  # Avoid division by zero
    normalized_distances = 1 - (np.array(distances) / max_distance)

    # Modify the calculation to take into account fire intensity and remaining suppressant.
    effective_suppression = np.minimum(
        np.array(fire_levels),
        (agent_fire_reduction_power * agent_suppressant_num) / np.array(fire_intensities)
    )
    
    # Calculate the effectiveness of the suppression
    suppression_effectiveness = effective_suppression / np.array(fire_levels)
    
    # Prevent over-evaluation of fires that can't be efficiently suppressed with remaining resources
    suppressant_sufficiency = agent_suppressant_num / np.array(fire_intensities)
    normalized_suppressant_sufficiency = suppressant_sufficiency / np.max(suppressant_sufficiency)

    # Calculate scores that take into account both suppression potential and 
    # distance importance increased (by increasing its exponent weight),
    # along with fire priority weights.
    temperature_distance = 0.3  # More shallow curve to prioritize closer fires
    temperature_effectiveness = 1.5  # Sharper curve to prioritize effective actions
    scores = (
        (normalized_distances ** temperature_distance) *
        (suppression_effectiveness ** temperature_effectiveness) *
        normalized_suppressant_sufficiency *
        np.array(fire_putout_weight)
    )

    # Selecting the fire with the maximum score
    selected_task_index = np.argmax(scores)

    return selected_task_index