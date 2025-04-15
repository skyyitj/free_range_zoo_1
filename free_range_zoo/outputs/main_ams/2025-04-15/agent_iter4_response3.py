-python
def single_agent_policy(
    agent_pos, agent_fire_reduction_power, agent_suppressant_num,
    other_agents_pos, fire_pos, fire_levels, fire_intensities, fire_putout_weight):

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    # Calculate the amount of fire an agent can put out
    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Adjusted temperature parameters
    level_temperature = 0.35
    intensity_temperature = 0.15
    distance_temperature = 0.01 

    for task in range(num_tasks):

        # get euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        # calculate score for each task using fire intensity, suppressant number and distance
        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            min(can_put_out_fire, fire_intensities[task]) * np.exp(-fire_intensities[task] * intensity_temperature)-
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]

    # return the index of the task with the highest score
    return np.argmax(scores)