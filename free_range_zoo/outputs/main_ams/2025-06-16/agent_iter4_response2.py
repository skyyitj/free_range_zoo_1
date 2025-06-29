def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
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

    # Define 'temperature' values for transformation
    distance_temperature = 10.0
    intensity_temperature = 15.0
    weight_temperature = 5.0

    num_tasks = len(fire_pos)
    scores = []

    for i in range(num_tasks):
        # Extract fire task properties
        fire_y, fire_x = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        putout_weight = fire_putout_weight[i]

        # Compute distance from agent to fire
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)

        # Transform components for scoring
        distance_score = np.exp(-distance / distance_temperature)  # Closer fires score higher
        intensity_score = np.exp(-fire_intensity / intensity_temperature)  # Lower intensity fires score higher
        weight_score = np.exp(putout_weight / weight_temperature)  # Higher priority fires score higher

        # Combine all components into a total score
        # The sign of the scores can be weighted differently to balance priorities
        score = distance_score + weight_score - intensity_score
        scores.append(score)

    # Select the fire task with the highest score
    best_task_index = np.argmax(scores)

    return best_task_index