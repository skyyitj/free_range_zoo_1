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

    num_tasks = len(fire_pos)

    scores = np.zeros(num_tasks)

    # Loop through all tasks to compute score for each
    for i in range(num_tasks):
        # Calculate the Euclidean distance from agent to fire
        distance = np.sqrt((fire_pos[i][0] - agent_pos[0]) ** 2 + (fire_pos[i][1] - agent_pos[1]) ** 2)
        # Introduce a temperature scale for distance
        temp_distance = 0.1
        weighted_distance = np.exp(-temp_distance * distance)

        # Potential reduction of fire intensity directly by this agent
        reduction_capability = agent_fire_reduction_power * agent_suppressant_num / (fire_intensities[i] + 0.001)

        # Introduce a temperature scale for effectiveness of reduction
        temp_effectiveness = 1.0
        weighted_effectiveness = np.exp(temp_effectiveness * reduction_capability)

        # Consider remaining fire after suppression
        remaining_intensity = max(0., fire_levels[i] - reduction_capability)

        # Priority based on remaining intensity: less is better, penalize high remains
        temp_priority = 0.2
        weighted_priority = np.exp(-temp_priority * remaining_intensity)

        # Prefer tasks with higher weights and within the reach of actual fire suppression
        scores[i] = fire_putout_weight[i] * weighted_distance * weighted_effectiveness * weighted_priority

    # Select the task with the highest score
    selected_task_index = np.argmax(scores)
    
    return selected_task_index