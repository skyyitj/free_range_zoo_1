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

    # Define temperature parameters for controlling impact of each component
    distance_temp = 1.0
    fire_intensity_temp = 1.0
    suppressant_efficiency_temp = 1.0
    reward_weight_temp = 1.0

    # Extract number of fire tasks
    num_tasks = len(fire_pos)

    # Function to calculate Euclidean distance
    def calculate_distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # Calculate scores for each fire task
    scores = []
    for i in range(num_tasks):
        # Calculate normalized distance (closer tasks preferred)
        distance = calculate_distance(agent_pos, fire_pos[i])
        distance_score = np.exp(-distance / distance_temp)

        # Normalize fire intensity (higher intensity fires prioritized)
        intensity_score = np.exp(fire_intensities[i] / fire_intensity_temp)

        # Calculate suppressant efficiency (ability to extinguish the fire)
        suppressant_efficiency = min(
            agent_fire_reduction_power * agent_suppressant_num, fire_intensities[i]
        )
        suppressant_efficiency_score = np.exp(
            suppressant_efficiency / suppressant_efficiency_temp
        )

        # Incorporate task reward weight
        reward_score = np.exp(fire_putout_weight[i] / reward_weight_temp)

        # Combine scores into a single value for prioritization
        score = (
            distance_score * intensity_score * suppressant_efficiency_score * reward_score
        )
        scores.append(score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(scores))

    return best_task_index