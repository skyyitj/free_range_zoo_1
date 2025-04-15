def single_agent_policy(agent_pos, agent_fire_power, agent_suppressant_num, other_agents_pos, fire_locations, fire_intensity_map):
    import numpy as np

    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SUPPRESS', 'WAIT']
    direction_vectors = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }

    def in_bounds(pos):
        return 0 <= pos[0] < fire_intensity_map.shape[0] and 0 <= pos[1] < fire_intensity_map.shape[1]

    best_score = -np.inf
    best_action = 'WAIT'

    # Attempt suppression if adjacent fire and suppressant available
    if agent_fire_power > 0 and agent_suppressant_num > 0:
        for direction, delta in direction_vectors.items():
            new_pos = (agent_pos[0] + delta[0], agent_pos[1] + delta[1])
            if in_bounds(new_pos) and new_pos in fire_locations:
                intensity = fire_intensity_map[new_pos]
                if intensity > 0:
                    return actions.index('SUPPRESS')

    # Move toward the highest-intensity fire cell (prioritize close, high-intensity)
    for direction, delta in direction_vectors.items():
        new_pos = (agent_pos[0] + delta[0], agent_pos[1] + delta[1])
        if in_bounds(new_pos):
            score = 0
            if new_pos in fire_locations:
                intensity = fire_intensity_map[new_pos]
                distance_penalty = 1  # Adjacent, so low penalty
                score = intensity / distance_penalty

            if score > best_score:
                best_score = score
                best_action = direction

    return actions.index(best_action)