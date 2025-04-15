def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int:
    
    #Get the distances to all fires from this agent
    distances = [np.sqrt((x-agent_pos[0])**2 + (y-agent_pos[1])**2) for x,y in fire_pos]
    #Get the distances to all fires from other agents
    other_distances = [np.sqrt((x-other_x)**2 + (y-other_y)**2) for other_x,other_y in other_agents_pos for x,y in fire_pos]

    fire_severity = np.multiply(fire_levels, fire_intensities)
    
    #Exclude fires being tackled by other agents by making their severity = 0
    for i in range(len(fire_pos)):
        if min(other_distances[i::len(fire_pos)]) < distances[i]:
            fire_severity[i] = 0

    #If there is no fire for this agent to tackle i.e. severities of all fires = 0, take a break and refill suppressant (return -1 for that)
    if max(fire_severity)==0: return -1

    #Else, find the fire with the highest severity
    return np.argmax(fire_severity)