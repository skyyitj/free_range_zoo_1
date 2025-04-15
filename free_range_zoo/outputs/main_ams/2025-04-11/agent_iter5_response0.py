def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float>>,
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    # Computing distances to fires
    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    
    # Defining reachable fires based on agent's suppressant resource
    reachable_fires = [i for i, dist in enumerate(distances) if dist <= agent_suppressant_num / agent_fire_reduction_power]

    if not reachable_fires:
        return -1
    
    # Incorporating distance and fire intensity in decision making
    reachable_fires.sort(key=lambda i: (distances[i], -fire_intensities[i]))
    chosen_fire = reachable_fires[0]  # Choose the closest fire with the highest intensity
    return chosen_fire