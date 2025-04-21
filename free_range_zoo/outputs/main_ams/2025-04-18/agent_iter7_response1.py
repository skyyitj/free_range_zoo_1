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

    max_score = -float('inf')
    best_fire = None

    # We increase the temperature for the distance and suppression power factors, giving more weight to the distance factor, 
    # and reducing the weight of suppression power.
    dist_temperature = 0.07
    power_temperature = 0.07

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # The suppression power now considers both the agent's reduction power and the available amount of suppressants, 
        # and is inversely proportional to the fire intensity.
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # The distance is now calculated with a Manhattan distance instead of Euclidean, which can more accurately reflect 
        # the grid based nature of the simulation environment.
        dist = abs(agent_pos[0]-fire_position[0]) + abs(agent_pos[1]-fire_position[1])

        # The score now considers both the fire weight and the suppression power, divided by the distance, 
        # ensuring that the agents prefer higher weight fires that are nearer and easier to put out.
        score = np.exp((fire_weight * suppression_power / (dist * dist_temperature + 1)) / power_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire