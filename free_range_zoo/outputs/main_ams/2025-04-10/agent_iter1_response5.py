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
    
    # If no suppressant is available, return -1 as no action can be performed
    if agent_suppressant_num <= 0:
        return -1

    # Priority calculation for each fire task
    fire_priorities = []
    for i, (fire_pos_i, fire_level_i, fire_intensity_i) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate the Euclidean distance between the agent and the fire
        distance_to_fire = ((agent_pos[0] - fire_pos_i[0]) ** 2 + (agent_pos[1] - fire_pos_i[1]) ** 2) ** 0.5
        
        # Calculate fire priority based on fire level, intensity, and proximity
        priority = (fire_level_i * 1.5 + fire_intensity_i * 2) / (distance_to_fire + 1)
        fire_priorities.append((i, priority))

    # Sort fires by priority (highest priority first)
    fire_priorities.sort(key=lambda x: x[1], reverse=True)

    # Check if any fire has a high enough level to naturally extinguish
    for i, priority in fire_priorities:
        if fire_levels[i] >= 10:  # Fire level of 10 means it will extinguish itself
            continue
        
        # If there's enough suppressant and the fire is still active, attempt to put it out
        if agent_suppressant_num > 0 and fire_levels[i] > 0:
            return i

    # If no valid fire task is found, return -1 (no action)
    return -1