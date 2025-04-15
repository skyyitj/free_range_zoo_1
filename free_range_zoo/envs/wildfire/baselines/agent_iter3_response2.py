def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    reachable_range = 5.0  # Define a range for agent
    validity_factor = 0.15  # Define a validity factor for fire level
    interference_factor = 0.1  # Define an interference factor for other agents

    if agent_suppressant_num <= 0.1:
        return -1

    valid_fires = [
        i for i, (fl, fi) in enumerate(zip(fire_levels, fire_intensities))
        if fl < fi * validity_factor
        and ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5 <= reachable_range
    ]

    if not valid_fires:
        return -1

    fire_scores = []
    for i in valid_fires:
        for other_agent_pos in other_agents_pos:
            other_to_fire_distance = ((other_agent_pos[0] - fire_pos[i][0])**2 + (other_agent_pos[1] - fire_pos[i][1])**2)**0.5
            agent_to_fire_distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
            
            if other_to_fire_distance < agent_to_fire_distance:
                interference = (1 / (other_to_fire_distance + 1)) * interference_factor
            else:
                interference = 0

            score = (agent_suppressant_num * fire_intensities[i] / (agent_to_fire_distance + 1)) * (1 - interference)
            fire_scores.append((score, i))

    task_to_address = max(fire_scores)[1]

    return task_to_address