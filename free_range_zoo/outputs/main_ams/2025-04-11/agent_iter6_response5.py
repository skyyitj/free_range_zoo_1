Python
def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
):
    # Divide the region into clusters
    clusters = cluster_regions(other_agents_pos + [agent_pos])
    # Assign the agent to the cluster which ic contains its position
    agent_cluster = assign_cluster(agent_pos, clusters)

    # Filter the fires that are in the same cluster as the agent
    fires_in_cluster = [i for i, pos in enumerate(fire_pos) if pos in clusters[agent_cluster]]
    fire_values = [fire_intensities[i] for i in fires_in_cluster]

    # Consider the distance of the fire from the agent
    agent_distance = [get_distance(agent_pos, fire_pos[i]) for i in fires_in_cluster]
    
    # Compute weights for each fire based on its intensity and proximity to the agent
    weights = [intensity / (distance**2 + 1) for intensity, distance in zip(fire_values, agent_distance)]
    
    # Select a fire using a stochastic policy (softmax over weights)
    fire_probabilities = softmax(weights)

    # If there's enough suppressant, choose the fire with the highest probability, otherwise recharge
    if agent_suppressant_num > np.max(fire_levels):
        chosen_fire = np.argmax(fire_probabilities)
    else:
        return -1  # indicate that the agent should recharge

    return chosen_fire