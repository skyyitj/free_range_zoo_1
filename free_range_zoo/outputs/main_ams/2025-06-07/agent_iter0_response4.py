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
    # How far is each fire from the agent
    import numpy as np
    distances = [
        np.sqrt((f_pos[0] - agent_pos[0]) ** 2 + (f_pos[1] - agent_pos[1]) ** 2)
        for f_pos in fire_pos
    ]

    # Score of each fire, considering both weight and distance
    scores = []
    for idx in range(len(fire_pos)):
        distance = distances[idx]
        # negative because smaller distance is better
        normalized_distance = np.exp(-distance / 5.0)  # temp constant 5.0
        
        fire_level = fire_levels[idx]
        intensity = fire_intensities[idx]
        weight = fire_putout_weight[idx]
        
        # We want larger reduction power and higher weight to be better
        
        # Suppression potential is the estimated reduction from our agent
        suppression_potential = min(intensity, agent_fire_reduction_power * agent_suppressant_num)
        normalized_suppression = suppression_potential / max(fire_intensities)  # normalize by max potential intensity to bring in [0,1]
        
        score = (weight * normalized_suppression * normalized_distance)

        # Suppressant use needing consideration
        suppressant_use_score = agent_fire_reduction_power / (agent_suppressant_num + 1e-6)  # avoid div by zero
        effective_score = score * np.exp(-suppressant_use_score / 10.0)  # temp constant 10.0

        scores.append(effective_score)

    # Select the highest score index
    selected_task_index = np.argmax(scores)
    return selected_task_index