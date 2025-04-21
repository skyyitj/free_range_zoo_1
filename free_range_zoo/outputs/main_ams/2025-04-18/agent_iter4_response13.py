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
    max_score = float('-inf')
    best_fire = None
    temperature = 0.01  # adjust the temperature parameter

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Modified distance-based factor for task selection
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1)

        # Modified firefighting efficiency factor in task selection
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)
      
        # Redesign the score calculation considering prioritization weight, firefighting efficiency, and distance factor
        # Redefine a soft-max function to adjust the score calculation mechanism to be more sensitive to both distance and fire weight. 
        # The shrinking parameter is revised in the scoring mechanism to better differentiate each task.
        weight_score = np.exp(fire_weight / temperature)
        suppression_score = np.exp(suppression_power / temperature)
        dist_score = np.exp(-dist / temperature)
        score = weight_score * suppression_score * dist_score

        # Introduce a penalty if the fire level is below a certain threshold
        if fire_level < 4:
            score -= penalty * fire_level
        
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire