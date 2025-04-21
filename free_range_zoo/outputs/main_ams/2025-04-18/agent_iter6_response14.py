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
  

    def distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

    best_score = -float('inf')
    best_fire = None
    normal_temperature = 0.02
    dist_temperature = 0.06  # Use a higher temperature to give more consideration to nearby fires
    level_temperature = 0.05  # Use a higher temperature to prevent high-level fires from burning out

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Calculate distance to the fire
        dist_to_fire = distance(agent_pos, fire_position)
        
        # Calculate distance to the nearest agent
        dist_to_nearest_agent = min(distance(agent_pos, other_agent_pos) for other_agent_pos in other_agents_pos)

        # Only the agent who is closest to the fire can put out the fire to make full use of the agent and avoid resource waste
        if dist_to_fire <= dist_to_nearest_agent:
            # Firefighting efficiency factor
            suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

            # Firefighting score
            score = np.exp((fire_weight * suppression_power / (dist_to_fire + 1) + fire_level) / normal_temperature)

            # Adjust the score according to distance, the nearer the fire, the more likely it would be selected 
            score *= np.exp((-dist_to_fire) / dist_temperature)
            # Adjust the score according to the fire level, the higher level the fire, the more urgently it needs to be put out
            score *= np.exp((fire_level) / level_temperature)

            # Check for new best score
            if score > best_score:
                best_score = score
                best_fire = i

    return best_fire