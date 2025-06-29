def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    # === Task Prioritization ===
    fire_putout_weight: List[float],
) -> int:
    # Ensure we have data for at least one fire
    if not fire_pos:
        raise ValueError("No fire positions provided.")
    
    # Helpful transformed variables
    num_tasks = len(fire_pos)
    
    # Initialize variables to store best task data
    best_task_index = -1
    max_score = float('-inf')
    
    # Characteristics of the fire task
    temp_intensity = 0.1  # Inverse temperature parameter for fire intensity
    
    for i in range(num_tasks):
        # Calculate Euclidean distance from the agent position to the fire position
        distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        
        # Transformations to create scores
        intensity_score = np.exp(-temp_intensity * fire_intensities[i])  # Smaller intensities are better
        weight_score = fire_putout_weight[i]  # Larger weights are better
        theoretical_suppression = agent_fire_reduction_power * agent_suppressant_num
        
        # Check if the agent can potentially extinguish this fire with its remaining suppressant
        if theoretical_suppression >= fire_levels[i]:
            suppressibility_score = 1.5  # gives a boost if the fire can be suppress completely
        else:
            suppressibility_score = 1.0
        
        # Calculate the score for this task
        score = ((weight_score * suppressibility_score) / (distance * 1.0) * intensity_score)
        
        # Update best task candidate
        if score > max_score:
            max_score = score
            best_task_index = i
    
    return best_task_index