def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    # Calculate the distances to each fire.
    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]

    # Find fires within reach.
    fires_within_reach = [i for i, dist in enumerate(distances) if dist <= agent_suppressant_num/agent_fire_reduction_power]

    # If no fires can be extinguished with the current amount of suppressant, the agent should recharge.
    if not fires_within_reach:
        return -1 

    # Calculate the expected suppressant usage of all fire tasks
    expected_suppressant_usage = [fire_levels[i] / agent_fire_reduction_power for i in fires_within_reach]

    # Calculate the "value" of each fire task: fire intensity / expected suppressant usage
    fire_values = [fire_intensities[i] / suppressant for i, suppressant in zip(fires_within_reach, expected_suppressant_usage)]

    # Choose the fire task with the highest adjusted value
    chosen_fire = max(range(len(fire_values)), key=fire_values.__getitem__)
    return chosen_fire