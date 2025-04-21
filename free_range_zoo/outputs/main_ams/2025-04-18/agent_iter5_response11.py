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
    
    # Initialize the maximum score and the selected fire index
    max_score = float('-inf')
    selected_fire_index = -1
    
    for i in range(len(fire_pos)):
        # Calculate the distance between the agent and the fire
        distance = ((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2) ** 0.5
        
        # Calculate the expected intensity reduction if this fire is selected
        expected_reduction = min(agent_fire_reduction_power * agent_suppressant_num, fire_levels[i])
        
        # Calculate the expected remaining fire intensity after suppression
        expected_remaining_intensity = fire_levels[i] - expected_reduction
        
        # Calculate the score for selecting this fire, considering the fire intensity,
        # distance to the fire, and the priority weight of the fire.
        # Moreover, a high score is given to fires that can be completely put out.
        score = (expected_reduction / max(1., distance)) * fire_putout_weight[i] * (1. + np.exp(-expected_remaining_intensity))
        
        # If the score for this fire is higher than the current maximum, update the maximum score 
        # and the index of the selected fire.
        if score > max_score:
            max_score = score
            selected_fire_index = i
            
    return selected_fire_index