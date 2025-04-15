def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:
    
    # If the agent's suppressant is low, recharge
    if agent_suppressant_num <= 0.2:
        return -2  # Return -2 to indicate need for recharge
    
    distances = np.array([np.sqrt((fire[0] - agent_pos[0])**2 + (fire[1] - agent_pos[1])**2) for fire in fire_pos])
    scores = np.array(fire_intensities) / distances
   
    # If fire level is above certain threshold, skip the task
    high_fire_levels = np.array(fire_levels) > 3
    scores[high_fire_levels] = np.inf

    for other_agent_pos in other_agents_pos:
        other_agent_distances = np.array([np.sqrt((fire[0] - other_agent_pos[0])**2 + (fire[1] - other_agent_pos[1])**2) for fire in fire_pos])
        close_to_other_agent = other_agent_distances < distances
        scores[close_to_other_agent] *= 2  # Increase the score if close to other agent
 
    return np.argmin(scores)  # Select the fire with the lowest score