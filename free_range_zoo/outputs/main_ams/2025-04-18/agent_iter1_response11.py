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

    max_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Calculate the distance between the agent and the fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5

        # Score = Weight / Distance² × Suppression Potential - Fire Intensity² 
        # This helps to prioritize fires that are closer and highly intense
        # It also considers agent's fire reduction ability, which can increase agent's effectiveness. 

        score = fire_weight / (dist*dist) * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - (fire_intensity*fire_intensity)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire