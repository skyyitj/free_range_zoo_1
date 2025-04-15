def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:
    
    num_fires = len(fire_pos)
    
    if agent_suppressant_num <= 0:
        return -1
    
    # Compute distances from the agent to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    fires_by_closer_agents = {i: [] for i in range(num_fires)}
    for other_agent_pos in other_agents_pos:
        # Compute distances from the other agent to all fires
        other_agent_distances = [np.sqrt((fire[0]-other_agent_pos[0])**2 + (fire[1]-other_agent_pos[1])**2) for fire in fire_pos]
        for i, (d1, d2) in enumerate(zip(distances, other_agent_distances)):
            if d2 < d1:
                fires_by_closer_agents[i].append(other_agent_pos)

    min_cost = np.inf
    min_cost_task = -1
    for task_id, closer_agents in fires_by_closer_agents.items():
        if closer_agents:
            continue
        cost = fire_intensities[task_id] * distances[task_id] / (agent_fire_reduction_power * agent_suppressant_num)
        if cost < min_cost:
            min_cost = cost
            min_cost_task = task_id

    if min_cost_task == -1:  # all fires are handled by closer agents
        for task_id in range(num_fires):
            cost = fire_intensities[task_id] * distances[task_id] / (agent_fire_reduction_power * agent_suppressant_num)
            if cost < min_cost:
                min_cost = cost
                min_cost_task = task_id

    return min_cost_task