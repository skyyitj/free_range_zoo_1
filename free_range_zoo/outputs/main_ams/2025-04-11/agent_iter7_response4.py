def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num <= 0:
        return -1

    # Calculate distances to all fires
    distances_to_fire = np.sqrt([np.sum(np.subtract(np.array(fire), np.array(agent_pos)) ** 2) for fire in fire_pos])

    # List of reachable fires (index, distance)
    reachable_fires = [(idx, distance) for idx, distance in enumerate(distances_to_fire) if distance <= agent_suppressant_num]

    if not reachable_fires:
        return -1

    # Calculate required suppressant for each fire
    suppressant_needed = [(fire_idx, fire_levels[fire_idx] / agent_fire_reduction_power) for fire_idx, _ in reachable_fires]

    # Calculate priority for each fire, prioritizing fires that require less suppressant and have higher intensity
    fire_priorities = [(fire_idx, suppressant/fire_intensities[fire_idx]) for fire_idx, suppressant in suppressant_needed]

    fire_priorities.sort(key=lambda x: x[1])
    chosen_fire = fire_priorities[0][0]
    return chosen_fire