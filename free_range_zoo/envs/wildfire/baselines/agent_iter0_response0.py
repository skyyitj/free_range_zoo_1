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
    task_scores = []

    # Evaluate each task
    for i in range(len(fire_pos)):
        task_pos = fire_pos[i]
        task_fire_level = fire_levels[i]
        task_fire_intensity = fire_intensities[i]

        # Calculate distance to task
        distance = (abs(agent_pos[0] - task_pos[0]) ** 2 + abs(agent_pos[1] - task_pos[1]) ** 2) ** 0.5

        # Check if other agents are closer to the task
        for other_agent_pos in other_agents_pos:
            other_agent_distance = (abs(other_agent_pos[0] - task_pos[0]) ** 2 + abs(other_agent_pos[1] - task_pos[1]) ** 2) ** 0.5
            if other_agent_distance < distance:
                # Skip this task, as other agents are closer and can handle it more effectively
                break
        else:
            # All other agents are farther away from this task, so consider it
            # Task score is a function of fire level, fire intensity, agent's fire reduction power, and available suppressant
            # For now, let's say our score function is: score = (fire_level - distance * fire_intensity) * (fire_reduction_power * available_suppressant)
            # Adjust the score function as needed, to reflect your specific strategy
            task_score = (task_fire_level - distance * task_fire_intensity) * (agent_fire_reduction_power * agent_suppressant_num)
            task_scores.append((i, task_score))

    if not task_scores:
        # No tasks to consider, choose a default action
        # For example, return -1 to indicate no action
        return -1
    else:
        # Choose the task with highest score
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores[0][0]