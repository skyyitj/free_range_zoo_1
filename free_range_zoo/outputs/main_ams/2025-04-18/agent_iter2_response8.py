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
        
        # Change: factor in the fire level in the distance calculation. More intense fires should appear closer.
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (1 + fire_level) / agent_suppressant_num

        # Change: add a factor that accounts for other agents' positions. We want to avoid all agents focusing on the same fire.
        other_agents_near = sum(1 / ((other_agent[0] - fire_position[0])**2 + (other_agent[1] - fire_position[1])**2 + 0.01)**0.5 for other_agent in other_agents_pos)
        
        # Change: account for other agents in the suppressant factor
        suppressant_factor = (agent_fire_reduction_power / (fire_intensity + other_agents_near)) * agent_suppressant_num
        
        score = fire_weight * suppressant_factor - dist * fire_intensity
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire