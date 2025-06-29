def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    # === Task Prioritization ===
    fire_putout_weight: List[float],
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import numpy as np

    num_tasks = len(fire_pos)
    task_scores = []

    # Temperature parameters for transformations
    intensity_temp = 0.5
    distance_temp = 1.0
    reward_temp = 1.5

    for i in range(num_tasks):
        # Distances between agent and fire location
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        
        # Normalize components
        fire_intensity_transformed = np.exp(fire_intensities[i] / intensity_temp)
        distance_transformed = np.exp(-distance / distance_temp)
        reward_weight_transformed = np.exp(fire_putout_weight[i] / reward_temp)

        # Compute score
        score = reward_weight_transformed * fire_intensity_transformed * distance_transformed

        # Penalize tasks if not enough suppressant to significantly reduce fire
        if agent_fire_reduction_power * agent_suppressant_num < fire_intensities[i]:
            score *= 0.5  # Reduce priority of tasks outside capability

        task_scores.append(score)

    # Select task with highest score
    best_task_idx = np.argmax(task_scores)
    return best_task_idx