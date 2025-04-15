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

    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    fires_within_reach = [i for i, dist in enumerate(distances) if dist <= agent_suppressant_num/agent_fire_reduction_power]

    if not fires_within_reach:
        # no fires can be completely extinguished with the current amount of suppressant, find the closest fire and partially extinguish it
        closest_fire = distances.index(min(distances))
        return closest_fire

    # find the fires that can be completely extinguished
    extinguishable_fires = [i for i in fires_within_reach if fire_levels[i] <= agent_fire_reduction_power * agent_suppressant_num]

    if extinguishable_fires:
        # There are fires that can be completely extinguished, choose the most intense one
        chosen_fire = max(extinguishable_fires, key=lambda i: fire_intensities[i])
    else:
        # no fires can be completely extinguished with the current amount of suppressant, find the closest fire and partially extinguish it
        chosen_fire = distances.index(min(distances))

    return chosen_fire