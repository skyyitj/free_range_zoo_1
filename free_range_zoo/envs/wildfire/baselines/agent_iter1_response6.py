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
        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...]
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...]
        fire_levels: Current fire level of each task
        fire_intensities: Intensity (difficulty) of each task

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1), or -1 if no valid task.
    """

    # If an agent has no suppressant, it can't act effectively, return -1 (no action)
    if agent_suppressant_num <= 0:
        return -1

    # Check if there are any fires to address
    if not fire_pos or len(fire_pos) == 0:
        return -1  # No fires to act on

    # Calculate the priority of each fire task based on fire level, intensity, and proximity to the agent
    fire_priorities = []
    for i, (fire_pos_i, fire_level_i, fire_intensity_i) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        # Calculate distance to the fire task
        distance_to_fire = ((agent_pos[0] - fire_pos_i[0]) ** 2 + (agent_pos[1] - fire_pos_i[1]) ** 2) ** 0.5
        
        # Calculate the "effort" required based on fire intensity, level, and distance
        # The higher the fire level, the higher the priority, but also the more intense the fire, the higher the urgency
        # Distance should make nearby fires more attractive
        priority = (fire_level_i * 1.5 + fire_intensity_i * 2) / (distance_to_fire + 1)

        fire_priorities.append((i, priority, fire_level_i, fire_intensity_i, distance_to_fire))

    # Sort fires by priority, highest priority first
    fire_priorities.sort(key=lambda x: x[1], reverse=True)

    # If there are no valid fire tasks or all fires are too intense (to be dealt with naturally), return -1
    for i, priority, fire_level, fire_intensity, distance in fire_priorities:
        if fire_level >= 10:  # Assuming a fire level of 10 means it will extinguish on its own
            # No need to address this fire, as it will burn out naturally
            continue

        # If we have enough suppressant and the fire is active (not yet extinguished), address it
        if agent_suppressant_num > 0 and fire_level > 0:
            return i  # Choose the fire task with the highest priority

    # If no fires are addressed, return -1 (no action)
    return -1