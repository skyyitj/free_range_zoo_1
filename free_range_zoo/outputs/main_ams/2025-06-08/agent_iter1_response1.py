def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y, x), ...]
    fire_levels: List[int],                      # Current fire intensity level of each fire
    fire_intensities: List[float],               # Current base difficulty of extinguishing each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Improved policy function to prioritize tasks optimally.
    """
    import math

    num_tasks = len(fire_pos)
    task_scores = []

    # Temperature parameters for score components
    distance_temp = 5.0
    fire_intensity_temp = 1.5  # Prioritize manageable fires
    priority_weight_temp = 0.8
    burnedout_temp = 2.0

    for i in range(num_tasks):
        # Calculate the Euclidean distance to fire location
        agent_y, agent_x = agent_pos
        fire_y, fire_x = fire_pos[i]
        distance = math.sqrt((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2)
        normalized_distance = math.exp(-distance / distance_temp)

        # Evaluate fire intensity for manageable suppression
        manageable_fire_intensity = fire_intensities[i] - agent_fire_reduction_power * agent_suppressant_num
        normalized_fire_intensity = math.exp(-manageable_fire_intensity / fire_intensity_temp)

        # Include task priority weights
        normalized_priority_weight = math.exp(fire_putout_weight[i] / priority_weight_temp)

        # Factor in fire intensity that is dangerously close to self-extinguish penalties
        burnedout_penalty = math.exp(-fire_intensities[i] / burnedout_temp) if fire_levels[i] >= 5 else 0

        # Combine scores using weighted summation
        task_score = (
            normalized_priority_weight * fire_levels[i]
            * (normalized_fire_intensity - burnedout_penalty)
            - normalized_distance
        )
        task_scores.append(task_score)

    # Select the task with the highest score while ensuring suppressant availability
    best_task_idx = max(
        range(num_tasks),
        key=lambda idx: task_scores[idx] if agent_suppressant_num > 0 else float('-inf')
    )

    return best_task_idx