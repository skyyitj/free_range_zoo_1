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

    # Initialize variables
    num_tasks = len(fire_pos)
    scores = []

    # Temperature parameters for normalization
    intensity_temp = 1.0     # Higher value reduces influence from intensity normalization
    weight_temp = 1.0        # Adjusts influence of priority weights
    distance_temp = 1.0      # Controls impact of distance normalization

    # Iterate over all fire tasks
    for i in range(num_tasks):
        # Fire intensity factor (tasks with higher fire intensity are prioritized)
        fire_intensity_factor = np.exp(fire_intensities[i] / intensity_temp)

        # Priority weight factor (fire-putout weight as provided by the system)
        priority_weight_factor = np.exp(fire_putout_weight[i] / weight_temp)

        # Distance factor (closer fire locations are more favorable)
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        distance_factor = np.exp(-distance / distance_temp)

        # Compute total score for the task
        score = (fire_intensity_factor * priority_weight_factor * distance_factor)

        # Penalize tasks if agent suppressant is insufficient for high-intensity fires
        if agent_suppressant_num * agent_fire_reduction_power < fire_intensities[i]:
            score *= 0.5  # Reduce score significantly but do not zero it

        # Add the score to the list
        scores.append(score)

    # Select the task with the maximum score
    selected_task = int(np.argmax(scores))

    return selected_task