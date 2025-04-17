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
    Determines the best action for an agent in the wildfire environment, optimizing suppressant efficiency.
    """

    # Initialize best task and its associated priority
    best_task_index = -1
    best_priority = -float('inf')

    # Iterate through each fire task to determine which one is best to address
    for i, fire_level in enumerate(fire_levels):
        fire_intensity = fire_intensities[i]
        
        # If the fire is about to extinguish naturally (high level), don't waste resources
        if fire_level >= 80:  # Arbitrary threshold for fire going out naturally
            continue

        # Calculate the distance to the fire
        fire_pos_y, fire_pos_x = fire_pos[i]
        agent_pos_y, agent_pos_x = agent_pos
        distance_to_fire = ((fire_pos_y - agent_pos_y) ** 2 + (fire_pos_x - agent_pos_x) ** 2) ** 0.5

        # If the agent doesn't have enough suppressant for this fire, skip it
        if agent_suppressant_num < fire_intensity:
            continue

        # Calculate potential suppression effectiveness
        suppression_effectiveness = agent_fire_reduction_power / (1 + distance_to_fire)

        # Calculate task priority: prioritize fires that can be suppressed more effectively
        task_priority = (suppression_effectiveness * (100 - fire_level)) / (1 + fire_intensity)

        # Weigh in the suppressant efficiency: prioritize maximizing suppression per unit of suppressant
        suppressant_efficiency = suppression_effectiveness / (1 + fire_intensity)
        task_priority *= suppressant_efficiency

        # If this task has a higher priority, update the best task
        if task_priority > best_priority:
            best_task_index = i
            best_priority = task_priority

    # Return the index of the fire with the highest priority
    return best_task_index