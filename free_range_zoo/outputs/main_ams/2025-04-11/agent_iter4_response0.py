import numpy as np  # Importing numpy

def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
):
    action_values = []

    for i in range(len(fire_pos)):
        dist_to_fire = np.sqrt((fire_pos[i][0]-agent_pos[0])**2 + (fire_pos[i][1]-agent_pos[1])**2)
        fire_suppression_value = agent_fire_reduction_power / fire_intensities[i]
        agent_on_same_fire = int(any([np.array_equal(fire_pos[i], other_pos) for other_pos in other_agents_pos]))

        action_val = fire_levels[i] * fire_suppression_value - dist_to_fire - 10*agent_on_same_fire

        if agent_suppressant_num <= 0 or fire_levels[i] <= 0:
            action_val = -1e8

        action_values.append(action_val)

    return np.argmax(action_values)