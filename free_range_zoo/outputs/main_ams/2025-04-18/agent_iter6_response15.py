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

    # === Scoring Fire Tasks ===
    max_score = -float('inf')
    best_fire = None
    temperature = 0.03  # Lower temperature to improve sensitivity of decision differentiation 

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance factor
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+0.5)

        # Firefighting efficiency factor 
        # where consideration for intensity has been made more significant by raising it to power 1.3
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity**1.3 + 0.5)

        # Score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # High fire level is given more weight by multiplying it with a factor of 1.5
        score = np.exp((fire_weight * suppression_power / (dist + 0.5) + fire_level*1.5) / temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire