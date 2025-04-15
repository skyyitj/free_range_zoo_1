def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    """
    Refined Decision Function for a Firefighting Agent.
    """

    def euclidean_distance(start, end):
        return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    # Adjust weights according to the agent performance feedback
    weight_level = 1.0
    weight_intensity = 4.0  # Increasing focus on intensity
    weight_distance = -1.0  # Negative as lower distance is preferred
    weight_other_agents_proximity = -0.2  # Decreased to lessen its impact

    best_fire_index = -1
    best_fire_score = float('-inf')

    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        distance_to_fire = euclidean_distance(agent_pos, pos)
        closer_agents_count = sum(1 for other_pos in other_agents_pos if euclidean_distance(other_pos, pos) < distance_to_fire)
        
        # If no suppressant is left, skip trying to tackle any fire.
        if agent_suppressant_num <= 0:
            continue
        
        # Incorporate suppressant potency relative to the intensity and remaining suppressant stock
        normalized_suppressant_intensity = agent_fire_reduction_power * agent_suppressant_num / (intensity + 1)

        # Calculate overall score for tackling this specific fire
        score = (weight_level * level +
                 weight_intensity * intensity +
                 weight_distance * distance_to_fire +
                 weight_other_agents_proximity * closer_agents_count +
                 normalized_suppressant_intensity)
        
        if score > best_fire_score:
            best_fire_score = score
            best_fire_index = i

    return best_fire_index