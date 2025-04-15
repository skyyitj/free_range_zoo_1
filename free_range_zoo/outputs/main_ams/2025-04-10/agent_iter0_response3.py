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

    # Step 1: Find the fires that are within range and are manageable (do not exceed agent's suppressant)
    manageable_fires = []
    for i, fire_pos_i in enumerate(fire_pos):
        distance_to_fire = ((fire_pos_i[0] - agent_pos[0])**2 + (fire_pos_i[1] - agent_pos[1])**2)**0.5
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]

        # The fire must be within reasonable range and the agent should have enough suppressant
        if distance_to_fire < 10:  # Assuming range limit to 10 units
            if agent_suppressant_num >= fire_intensity:
                manageable_fires.append(i)

    # Step 2: If there are manageable fires, prioritize based on intensity (highest priority first)
    if manageable_fires:
        best_fire = max(manageable_fires, key=lambda idx: fire_intensities[idx])
        return best_fire

    # Step 3: If no manageable fire found, check if any fire is about to extinguish naturally
    # We avoid letting fires reach a high enough level, where they go out naturally.
    for i, fire_level in enumerate(fire_levels):
        if fire_level >= 8:  # Fire level threshold for natural extinguishment
            return i

    # If no other suitable action is found, return -1 (no action, agent waits)
    return -1