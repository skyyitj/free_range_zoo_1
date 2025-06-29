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

    # Temperature parameters for tuning transformation sensitivity
    distance_temperature = 0.1
    intensity_temperature = 0.5
    suppressant_temperature = 2.0
    conservation_temperature = 3.0

    # Calculate distances of the agent to each fire
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Normalize distances and transform to prioritize nearby fires more aggressively
    normalized_distances = 1 / (np.array(distances) + 0.001)  # smooth division
    distance_scores = np.exp(normalized_distances / distance_temperature)  # apply temperature to distance

    # Calculate potential suppressant effect on each fire
    potential_fire_reductions = agent_supressant_num * agent_fire_reduction_power / np.array(fire_intensities)
    
    # Calculate normalized fire levels where taking account of the fire intensities
    normalized_fire_levels = np.array(fire_levels) / np.array(fire_intensities)
    fire_level_scores = np.exp(normalized_fire_levels / intensity_temperature)  # intensified sensitivity to hot fires

    # Compute suppressant conservation effect
    suppressant_consumed = np.clip(potential_fire_reductions, 0, agent_supressant_num)
    suppressant_left = agent_supressant_num - suppressant_consumed
    suppressant_conservation_scores = np.exp((suppressant_left / agent_supressant_num) / suppressant_temperature)

    # Calculate how desperate a fire needs to be addressed based on current intensity and potential impact after a step
    post_suppression_fire_levels = np.clip(normalized_fire_levels - potential_fire_reductions, 0, None)
    desperation_scores = (post_suppression_fire_levels == 0).astype(float)
    desperation_scores = np.exp(desperation_scores / conservation_temperature)  # priority to finishing jobs

    # Combine all scores to determine which fire to target
    # Combine using multiplication of relevant scores
    combined_scores = (
        distance_scores *
        fire_level_scores *
        np.array(fire_putout_weight) *  # directly use priority weights
        suppressant_conservation_scores *
        desperation_scores
    )
    
    # Selection of the fire task with the highest score
    selected_task_index = np.argmax(combined_scores)

    return selected_task_index