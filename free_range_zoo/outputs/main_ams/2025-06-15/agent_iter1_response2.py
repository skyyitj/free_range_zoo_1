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
    import numpy as np
    
    num_tasks = len(fire_pos)
    task_scores = []

    # Distance scaling temperature
    distance_temp = 1.0
    intensity_temp = 1.5
    reward_temp = 1.0

    for i in range(num_tasks):
        # Distance between agent and fire task
        agent_y, agent_x = agent_pos
        fire_y, fire_x = fire_pos[i]
        distance = np.sqrt((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2)

        # Distance penalty (lower distance increases preference)
        distance_factor = np.exp(-distance / distance_temp)

        # Fire intensity factor (higher intensity increases preference)
        intensity_factor = np.exp(fire_intensities[i] / intensity_temp)

        # Reward weight factor (higher weight increases preference)
        reward_factor = np.exp(fire_putout_weight[i] / reward_temp)

        # Estimate fire suppression outcome
        expected_reduction = agent_fire_reduction_power * agent_suppressant_num
        remaining_fire = max(0, fire_intensities[i] - expected_reduction)

        # Remaining fire penalty (fires closer to 0 are preferable)
        remaining_fire_penalty = 1 / (remaining_fire + 1e-6)

        # Calculate task score as a weighted combination of factors
        task_score = reward_factor * intensity_factor * distance_factor * remaining_fire_penalty
        task_scores.append(task_score)

    # Select the task with the maximum score
    return int(np.argmax(task_scores))