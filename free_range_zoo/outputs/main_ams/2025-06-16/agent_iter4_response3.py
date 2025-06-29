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

    num_tasks = len(fire_pos)
    task_scores = []

    # Temperature parameters for score components
    intensity_temperature = 0.5
    distance_temperature = 1.5
    reward_temperature = 1.0

    for i in range(num_tasks):
        # Fire properties
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        fire_weight = fire_putout_weight[i]
        fire_location = fire_pos[i]

        # Compute distance from agent to fire
        distance_to_fire = np.linalg.norm([agent_pos[0] - fire_location[0], agent_pos[1] - fire_location[1]])

        # Score components
        intensity_score = np.exp(-fire_intensity / intensity_temperature)  # Higher score for lower intensity
        distance_score = np.exp(-distance_to_fire / distance_temperature)  # Higher score for closer fires
        reward_score = np.exp(fire_weight / reward_temperature)  # Higher score for higher task reward weight

        # Combine scores with normalization
        total_score = reward_score * intensity_score * distance_score
        task_scores.append(total_score)

    # Select the fire task with the highest score
    optimal_task_index = int(np.argmax(task_scores))

    return optimal_task_index