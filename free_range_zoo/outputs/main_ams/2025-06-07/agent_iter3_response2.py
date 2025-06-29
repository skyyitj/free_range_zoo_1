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
    max_distance = np.max(distances) if np.max(distances) else 1  # Avoid division by zero
    normalized_distances = 1 - (np.array(distances) / max_distance)

    # Calculate the potential reduction in fire intensity based on agent's action
    potential_impact = agent_fire_reduction_power * agent_supressant_num / np.array(fire_intensities)
    
    # Create a metric which combines all necessary considerations
    # Including the benefit of potentially putting out the fire, weighted priority and the distance weight
    scores = (
        potential_impact * np.array(fire_levels) * np.array(fire_putout_weight) * normalized_distances
    )
    
    # Selecting the fire with the maximum score
    selected_task_index = np.argmax(scores)

    return selected_task_index