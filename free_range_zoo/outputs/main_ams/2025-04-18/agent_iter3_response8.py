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

    max_score = -float("inf")
    best_fire = None

    for i in range(len(fire_pos)):
        # Factor in the distance from agent to fire location
        dist = ((agent_pos[0]-fire_pos[i][0])**2 + (agent_pos[1]-fire_pos[i][1])**2)**0.5
        distance_factor = 1 / (dist + 1)

        # Factor in the intensity of fire and the agent's fire reduction power
        expected_suppress = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i])
        suppress_factor = expected_suppress / (fire_intensities[i] + 1)

        # Factor in the risks of fire spread and the relative priority of different fire tasks
        risk_factor = fire_levels[i] * fire_putout_weight[i]

        # Calculate the comprehensive score for each task
        score = distance_factor * suppress_factor * risk_factor

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire