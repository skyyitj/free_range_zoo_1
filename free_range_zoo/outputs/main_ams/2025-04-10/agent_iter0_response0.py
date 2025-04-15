def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
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
    
    # Priority 1: If a fire is highly intense (e.g., fire level > threshold), prioritize it
    high_priority_fires = [
        i for i, level in enumerate(fire_levels) if level > 0.7 and fire_intensities[i] > 0.5
    ]
    
    # Priority 2: If suppressant resources are enough, choose the fire with the highest intensity
    if agent_suppressant_num > 0.3:
        # Find fire with the highest intensity that hasn't burned out
        highest_intensity_fire_idx = max(range(len(fire_intensities)), key=lambda i: fire_intensities[i])
        return highest_intensity_fire_idx

    # Priority 3: If there's enough suppressant, choose the most urgent fire task
    if len(high_priority_fires) > 0:
        return high_priority_fires[0]

    # Priority 4: If not enough suppressant, consider the fire levels as an indicator of urgency
    lowest_fire_level_idx = min(range(len(fire_levels)), key=lambda i: fire_levels[i])
    
    # Avoid fires that are about to extinguish naturally (level > threshold)
    if fire_levels[lowest_fire_level_idx] > 0.7:
        return lowest_fire_level_idx

    return lowest_fire_level_idx