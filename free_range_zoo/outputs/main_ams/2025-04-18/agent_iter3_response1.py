def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float, 

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int],  
    fire_intensities: List[float], 

    fire_putout_weight: List[float]
) -> int:

    agent_pos_array = np.array(agent_pos)

    # Calculate distances to each fire
    distances = np.linalg.norm(np.array(fire_pos) - agent_pos_array, axis=1)

    # Compute the fire reduction ability of the agent for each fire
    fire_reductions = agent_fire_reduction_power * agent_suppressant_num / (np.array(fire_intensities)+1)

    # Compute the score (value) gained by fighting each fire
    fire_values = np.array(fire_putout_weight) * fire_reductions / (distances + 1)

    # Calculate the cost of the fire (based on its intensity and distance)
    fire_costs = distances * (np.array(fire_levels) ** 2)

    # Calculate the net value of choosing each fire (value - cost)
    net_values = fire_values - fire_costs

    # Choose the fire with the highest net value
    chosen_fire = np.argmax(net_values)

    return chosen_fire