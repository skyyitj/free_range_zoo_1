def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight,
):
    # Calculate distance to each fire
    distances = np.linalg.norm(np.array(fire_pos) - np.array(agent_pos), axis=1)

    # Choose the closest fire
    chosen_fire = np.argmin(distances)
    
    return chosen_fire