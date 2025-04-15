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
    """
    # If there's no suppressant, the agent cannot act, so return -1 (no task to handle)
    if agent_suppressant_num <= 0:
        return -1
    
    # Find fires that are still active (not naturally extinguished)
    active_fires = [
        i for i, fire_level in enumerate(fire_levels) if fire_level < 1.0
    ]
    
    # If there are no active fires, return -1 (no task)
    if not active_fires:
        return -1
    
    # Determine which fire to prioritize
    best_fire_index = None
    best_fire_priority = float('inf')

    for i in active_fires:
        # Calculate how much suppressant is needed to reduce this fire
        fire_intensity = fire_intensities[i]
        required_suppressant = fire_intensity / agent_fire_reduction_power

        # If the agent does not have enough suppressant for this fire, skip it
        if required_suppressant > agent_suppressant_num:
            continue

        # Calculate the distance between the agent and the fire (Euclidean distance)
        fire_position = fire_pos[i]
        distance = ((agent_pos[0] - fire_position[0]) ** 2 + (agent_pos[1] - fire_position[1]) ** 2) ** 0.5
        
        # Fire priority is determined by a combination of fire level (higher = more urgent) and distance (closer = more urgent)
        fire_priority = fire_levels[i] / (distance + 1e-6)

        # Check if this fire has a higher priority
        if fire_priority < best_fire_priority:
            best_fire_priority = fire_priority
            best_fire_index = i

    # If there's a fire that can be tackled, return the best fire index
    if best_fire_index is not None:
        return best_fire_index

    # If no fire can be tackled, return -1
    return -1