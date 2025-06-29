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
    """
    import numpy as np

    num_tasks = len(fire_pos)
    if num_tasks == 0:
        return -1  # No tasks available, return invalid task index

    scores = []

    # Temperature parameters for score component scaling
    distance_temp = 2.0
    intensity_temp = 5.0
    priority_temp = 3.0

    # Iterate through each fire task to compute a score
    for i in range(num_tasks):
        # Compute distance from agent to fire location
        dist = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)

        # Normalized distance score (smaller distance is better)
        distance_score = np.exp(-dist / distance_temp)

        # Fire intensity score (higher intensity is more urgent)
        intensity_score = np.exp(fire_intensities[i] / intensity_temp)

        # Task priority weight (direct impact of priority weights)
        priority_score = np.exp(fire_putout_weight[i] / priority_temp)

        # Resource availability check: Avoid assigning to tasks the agent cannot meaningfully impact
        if agent_suppressant_num > 0:
            effective_reduction = agent_fire_reduction_power * agent_suppressant_num
            remaining_fire = fire_intensities[i] - effective_reduction
        else:
            remaining_fire = fire_intensities[i]  # No suppressant left, fire remains unchanged

        # Discourage assigning tasks where fire cannot be extinguished soon
        potential_extinguish_score = 1.0 if remaining_fire <= 0 else 1.0 / (1 + remaining_fire)

        # Combine scores (weight them for final decision)
        score = (
            1.5 * distance_score +  # Weighting for distance
            2.0 * intensity_score +  # Weighting for fire intensity
            2.5 * priority_score +  # Weighting for priority of the fire task
            2.0 * potential_extinguish_score  # Weighting for extinguishability
        )
        scores.append(score)

    # Select the task with the highest score
    chosen_task_index = int(np.argmax(scores))
    return chosen_task_index