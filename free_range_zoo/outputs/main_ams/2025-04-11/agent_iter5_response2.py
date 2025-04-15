def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + 
                         (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    
    fires_within_reach = [i for i, dist in enumerate(distances) if dist <= agent_suppressant_num/agent_fire_reduction_power]

    if not fires_within_reach:
        return -1

    # Determine if there are other agents closer to the fires and if they have more suppressant
    closest_agents_to_fires = [np.argmin([np.sqrt((fire[0] - agent[0]) ** 2 + 
                                               (fire[1] - agent[1]) ** 2) for agent in other_agents_pos]) for fire in fire_pos]

    fires_in_need = [fire for fire in fires_within_reach if 
                     closest_agents_to_fires[fire] != agent_pos or 
                     (closest_agents_to_fires[fire] == agent_pos and agent_suppressant_num > other_agents_pos[closest_agents_to_fires[fire]][2])]

    if not fires_in_need:
        return -1

    # Choose the fire with the highest intensity that is still in need
    chosen_fire = max(fires_in_need, key=lambda i: (fire_levels[i], fire_intensities[i]))
    return chosen_fire