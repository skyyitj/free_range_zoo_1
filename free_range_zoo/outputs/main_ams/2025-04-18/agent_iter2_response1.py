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
    # === Fire Task Selection Scoring ===
    max_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Add a factor of (agent_suppressant_num/fire_intensity) in the distance calculation 
        # to prioritize fires that can be effectively controlled by the agent's suppressant.
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num / fire_intensity)
        
        # Modify the scoring formula to improve the agents' efficiency in suppressant usage by 
        # dividing the fire_weight by the number of agents that are not currently handling any fire.
        idle_agents = len([pos for pos in other_agents_pos if pos not in set(fire_pos)])
        if idle_agents == 0: idle_agents = 1  # avoid division by zero
        score = (fire_weight / idle_agents) - dist
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire