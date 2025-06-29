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

    # Initialize decision variables
    num_tasks = len(fire_pos)
    task_scores = [0] * num_tasks

    # Temperature parameters for component scaling
    intensity_temp = 2.0
    distance_temp = 10.0
    priority_temp = 1.0

    # Iterate over all tasks and compute a score for each
    for i in range(num_tasks):
        # Fire properties
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        fire_priority_weight = fire_putout_weight[i]
        fire_position = fire_pos[i]

        # Compute distance between agent and fire location
        distance = np.linalg.norm([agent_pos[0] - fire_position[0], agent_pos[1] - fire_position[1]])

        # Compute expected suppression potential
        expected_suppression = min(agent_suppressant_num, fire_intensity) * agent_fire_reduction_power
        remaining_fire = max(0, fire_intensity - expected_suppression)

        # Score components
        # A. Intensity contribution: Fires with high intensity have higher urgency
        intensity_contribution = np.exp(fire_intensity / intensity_temp)

        # B. Distance contribution: Fires closer to the agent are prioritized
        distance_contribution = np.exp(-distance / distance_temp)

        # C. Priority weight contribution: Incorporate predefined priority weight
        priority_contribution = np.exp(fire_priority_weight / priority_temp)

        # D. Remaining fire penalty: Prefer fires with lower expected remaining intensity after suppression
        remaining_fire_penalty = np.exp(-remaining_fire / intensity_temp)

        # Combine weighted contributions to compute task score
        task_scores[i] = (
            intensity_contribution * priority_contribution * distance_contribution * remaining_fire_penalty
        )

    # Select the task with the highest score
    optimal_task_index = int(np.argmax(task_scores))
    return optimal_task_index