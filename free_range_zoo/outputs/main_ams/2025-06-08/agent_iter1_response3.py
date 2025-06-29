def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                     # Current intensity level of each fire
    fire_intensities: List[float],              # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],            # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Input Parameters:
        See above.

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    import math

    num_tasks = len(fire_pos)
    task_scores = []

    # Modified temperature parameters
    distance_temp = 3.0
    fire_intensity_temp = 1.5
    suppression_efficiency_temp = 2.0
    priority_weight_temp = 1.0

    for i in range(num_tasks):
        # Calculate the Euclidean distance to fire location
        agent_y, agent_x = agent_pos
        fire_y, fire_x = fire_pos[i]
        distance = math.sqrt((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2)
        normalized_distance = math.exp(-distance / distance_temp)

        # Evaluate fire intensity impact (higher priority for higher intensity fires)
        normalized_fire_intensity = math.exp(fire_intensities[i] / fire_intensity_temp)

        # Include suppressant efficiency as a priority factor
        suppressant_efficiency_score = (
            agent_fire_reduction_power / (fire_intensities[i] + 1)  # Avoid division by zero
        )
        normalized_suppressant_efficiency = math.exp(suppressant_efficiency_score / suppression_efficiency_temp)

        # Account for task priority weights
        normalized_priority_weight = math.exp(fire_putout_weight[i] / priority_weight_temp)

        # Combine scores using weighted summation (higher weights for intensity and priority)
        task_score = (
            normalized_priority_weight * normalized_fire_intensity
            - normalized_distance
            + normalized_suppressant_efficiency
        )

        task_scores.append(task_score)

    # Select the task with the highest score while checking suppressant availability
    best_task_idx = max(
        range(num_tasks),
        key=lambda idx: task_scores[idx] if agent_suppressant_num > 0 else float('-inf')
    )

    return best_task_idx