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

    # Put the distance, intensity and index of each fire in a list
    fires_info = []
    for idx, (pos, lvl, inten) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        if lvl > agent_fire_reduction_power*agent_suppressant_num:  # Not enough suppressant for this fire
            continue
        dist = ((agent_pos[0] - pos[0])**2 + (agent_pos[1] - pos[1])**2)**0.5
        fires_info.append((dist, -inten, idx))  # Use negative intensity to prioritize high intensity fires

    # If no fires are in range or suppressant is not enough, the agent should recharge
    if not fires_info:
        return -1  # -1 will indicate the agent needs to recharge

    # Sort the fires by distance and then by intensity (prioritize closer, more intense fires)
    fires_info.sort()

    return fires_info[0][2]  # Return the index of the chosen fire