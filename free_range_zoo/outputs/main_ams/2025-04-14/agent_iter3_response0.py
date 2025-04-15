def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num <= 0.1:
        return -1

    valid_fires = [i for i, fl in enumerate(fire_levels) if fl < fire_intensities[i]]

    if not valid_fires:
        return -1

    fire_scores = []
    for i in valid_fires:
        distance_self = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        other_agents_distances = [((agent_pos[0] - pos[0])**2 + (agent_pos[1] - pos[1])**2)**0.5 for pos in other_agents_pos]
        closest_other_agent_distance = min(other_agents_distances) if other_agents_distances else 0
        score = (agent_suppressant_num * fire_intensities[i]) / ((distance_self + 1) * (1 + closest_other_agent_distance))
        fire_scores.append((score, i))

    task_to_address = max(fire_scores)[1]

    return task_to_address