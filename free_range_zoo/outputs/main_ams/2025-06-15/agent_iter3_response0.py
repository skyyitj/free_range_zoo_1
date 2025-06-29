def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    import numpy as np

    # Temperature parameters for score components
    intensity_temperature = 1.0  # For fire intensities
    distance_temperature = 1.0  # For distances
    weight_temperature = 1.0  # For prioritization weights

    # Initialize task scores
    task_scores = []

    # Calculate scores for each task
    for i in range(len(fire_pos)):
        fire_loc = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        putout_weight = fire_putout_weight[i]

        # Calculate the distance to the fire
        agent_y, agent_x = agent_pos
        fire_y, fire_x = fire_loc
        distance = np.sqrt((agent_y - fire_y)**2 + (agent_x - fire_x)**2)

        # Normalize and transform the distance using temperature
        distance_score = np.exp(-distance / distance_temperature)

        # Normalize and transform the fire intensity using temperature
        intensity_score = np.exp(-fire_intensity / intensity_temperature)

        # Transform reward weight using temperature
        weight_score = np.exp(putout_weight / weight_temperature)

        # Combine scores: prioritize fire level, distance, and reward weight
        combined_score = (weight_score * distance_score) / (intensity_score + fire_level)

        task_scores.append(combined_score)

    # Select the task with the maximum score
    optimal_task_index = np.argmax(task_scores)

    return optimal_task_index