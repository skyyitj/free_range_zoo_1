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

    # Number of fires to evaluate
    num_tasks = len(fire_pos)

    # Temperature values for score normalization
    distance_temperature = 1.0  # Higher means less sensitivity to distance
    intensity_temperature = 1.0  # Higher means less sensitivity to fire intensity
    weight_temperature = 1.0  # Higher means less sensitivity to fire weight

    # Helper function: Euclidean distance
    def euclidean_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Scoring each fire task
    scores = []
    for i in range(num_tasks):
        distance = euclidean_distance(agent_pos, fire_pos[i])
        normalized_distance = np.exp(-distance / distance_temperature)  # Normalize distance

        intensity = fire_intensities[i]
        normalized_intensity = np.exp(-intensity / intensity_temperature)  # Normalize intensity

        weight = fire_putout_weight[i]
        normalized_weight = np.exp(weight / weight_temperature)  # Normalize weight

        # Calculate score considering distance, intensity, and weight
        score = normalized_distance * normalized_intensity * normalized_weight
        scores.append(score)

    # Choose the task with the highest score
    selected_task = np.argmax(scores)
    return selected_task