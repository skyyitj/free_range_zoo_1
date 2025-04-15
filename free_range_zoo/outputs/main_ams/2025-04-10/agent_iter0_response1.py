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
    Determines the best action for an agent in the wildfire environment.

    Args:
        agent_pos: Position of this agent (y, x)
        agent_fire_reduction_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available

        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...] shape: (num_agents-1, 2)

        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...] shape: (num_tasks, 2)
        fire_levels: Current fire level of each task shape: (num_tasks,)
        fire_intensities: Intensity (difficulty) of each task shape: (num_tasks,)

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    
    # We will prioritize fires that are within the agent's suppressant capacity
    best_task_index = -1
    max_fire_reduction = -float('inf')
    
    # Check each fire task
    for i, (fire_pos_i, fire_level, fire_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate distance to the fire (Euclidean distance is assumed)
        distance_to_fire = ((fire_pos_i[0] - agent_pos[0]) ** 2 + (fire_pos_i[1] - agent_pos[1]) ** 2) ** 0.5
        
        # If fire level is high enough, it may extinguish by itself (no action taken)
        if fire_level > 0.8 and fire_intensity > 0.6:
            continue
        
        # Check if the agent has enough suppressant and fire reduction power for this fire
        fire_reduction_effectiveness = min(agent_fire_reduction_power, fire_level) * (1 / (1 + distance_to_fire))
        
        # Factor in the remaining suppressant available
        suppressant_needed = fire_intensity * distance_to_fire
        if agent_suppressant_num >= suppressant_needed:
            # Choose the task where the agent can reduce the most fire level
            if fire_reduction_effectiveness > max_fire_reduction:
                max_fire_reduction = fire_reduction_effectiveness
                best_task_index = i
    
    # If no fire task is available, return a default (could be any fire index)
    if best_task_index == -1:
        return 0
    
    return best_task_index