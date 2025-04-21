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
    highest_score = float('-inf')
    best_fire = None

    for i, (fire_position, fire_intensity_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate the distance of the agent from the fire position using Euclidean distance
        distance = ((agent_pos[0]-fire_position[0])**2 + (agent_pos[1]-fire_position[1])**2)**0.5
        
        # Increase the weight for fire that is closer to the agent and has higher intensity
        # Factor in the available suppressant with the agent fire reduction power to prioritize their actions
        intensity_factor = ((agent_fire_reduction_power/fire_intensity) * agent_suppressant_num) / distance
        score = fire_weight * intensity_factor - fire_intensity_level

        # If computed score for a fire is higher than the highest_score, update the highest_score and best_fire
        if score > highest_score:
            highest_score = score
            best_fire = i

    return best_fire