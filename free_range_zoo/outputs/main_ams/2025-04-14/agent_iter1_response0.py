def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num <= 0:  # If the agent has no suppressant left, return -1 to recharge.
        return -1

    valid_fires = [i for i, fl in enumerate(fire_levels) if fl < fire_intensities[i]]  # Tasks that aren't self-extinguishing

    if not valid_fires:  # If all fires are self-extinguishing, return -1.
        return -1

    # Calculate scores for each fire. Higher score means higher priority. A fire's score is inversely proportional to its distance to the agent and directly proportional to its intensity. Fires surrounded by many other fires are penalized.
    fire_scores = []
    for i in valid_fires:
        # Calculate the Euclidean distance from the agent to the fire.
        distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5

        # Calculate the number of other fires within a certain radius.
        nearby_fires = sum(1 for j in valid_fires if ((fire_pos[i][0] - fire_pos[j][0])**2 + (fire_pos[i][1] - fire_pos[j][1])**2)**0.5 <= 3)

        # Calculate the fire's score.
        score = fire_intensities[i] / ((distance + 1) * (nearby_fires + 1))
        fire_scores.append((score, i))

    # Choose the fire with the highest score.
    task_to_address = max(fire_scores)[1]

    return task_to_address