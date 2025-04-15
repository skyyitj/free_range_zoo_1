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

    # If no suppressant, cannot take action
    if agent_suppressant_num <= 0:
        return -1

    # If no fires to address, return -1
    if not fire_pos:
        return -1

    # Calculate priority for each fire task based on fire level, intensity, and distance
    fire_priorities = []
    for i, (fire_pos_i, fire_level_i, fire_intensity_i) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate distance to the fire task
        distance_to_fire = ((agent_pos[0] - fire_pos_i[0]) ** 2 + (agent_pos[1] - fire_pos_i[1]) ** 2) ** 0.5
        
        # Calculate a priority score based on fire level, intensity, and distance
        priority = (fire_level_i * 1.5 + fire_intensity_i * 2) / (distance_to_fire + 1)
        fire_priorities.append((i, priority, fire_level_i, fire_intensity_i, distance_to_fire))

    # Sort fires by priority, highest first
    fire_priorities.sort(key=lambda x: x[1], reverse=True)

    # Attempt to extinguish the most prioritized fire
    for i, priority, fire_level, fire_intensity, distance in fire_priorities:
        # If fire level is high enough, skip this task (it will extinguish on its own)
        if fire_level >= 10:  # Assuming level 10 means the fire extinguishes itself
            continue
        
        # If suppressant is available and the fire needs suppression
        if agent_suppressant_num > 0 and fire_level > 0:
            return i  # Return the index of the most important fire to address

    # If no valid fire to extinguish, return -1
    return -1