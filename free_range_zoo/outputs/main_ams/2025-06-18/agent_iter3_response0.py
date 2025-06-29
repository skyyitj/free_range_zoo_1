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
    fire_putout_weight: List[float],             
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    num_tasks = len(fire_pos)
    task_scores = []

    # Temperature parameters for score components
    intensity_temp = 5.0  # Temperature for fire intensity normalization
    distance_temp = 10.0  # Temperature for distance normalization
    priority_temp = 3.0   # Temperature for reward prioritization

    # Iterate over all fire tasks and calculate a score for each
    for i in range(num_tasks):
        fire_position = fire_pos[i]

        # Distance from agent to the fire location
        distance = ((agent_pos[0] - fire_position[0]) ** 2 + (agent_pos[1] - fire_position[1]) ** 2) ** 0.5
        
        # Calculate effective reduction of fire intensity if this fire is targeted
        effective_intensity_reduction = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i])

        # Remaining fire intensity after suppression
        remaining_fire_intensity = fire_intensities[i] - effective_intensity_reduction

        # Normalize components of the score
        normalized_intensity = np.exp(-remaining_fire_intensity / intensity_temp)
        normalized_distance = np.exp(-distance / distance_temp)
        normalized_priority = np.exp(fire_putout_weight[i] / priority_temp)

        # Composite score combining intensity, distance, and task priority
        score = normalized_intensity * normalized_priority * normalized_distance
        task_scores.append(score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(task_scores))
    return best_task_index