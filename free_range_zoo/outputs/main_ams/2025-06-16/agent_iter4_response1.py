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
    scores = []

    # Temperature values for score component scaling
    intensity_temp = 1.0
    suppressant_temp = 1.0
    proximity_temp = 0.5

    for i in range(num_tasks):
        # Fire position and intensity information
        fire_y, fire_x = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_priority = fire_putout_weight[i]
        fire_level = fire_levels[i]

        # Remaining fire intensity after agent suppression
        remaining_intensity = fire_intensity - (agent_suppressant_num * agent_fire_reduction_power)

        # Score component 1: Prioritize tasks with higher fire intensity
        intensity_score = np.exp(fire_intensity / intensity_temp)

        # Score component 2: Penalize resource usage if suppressant is limited
        suppressant_score = np.exp(-agent_suppressant_num / suppressant_temp)

        # Score component 3: Avoid assigning tasks close to other agents
        agent_proximity = sum(np.exp(-np.linalg.norm(np.array(agent_pos) - np.array(other_pos)) / proximity_temp)
                              for other_pos in other_agents_pos)
        proximity_score = 1 / (1 + agent_proximity)  # Inverse scaling for proximity

        # Score component 4: Account for task priority weight
        priority_score = fire_priority

        # Aggregate score with weights
        aggregated_score = priority_score * (intensity_score - suppressant_score + proximity_score)

        # Avoid assigning tasks where fire might self-extinguish with penalty
        if fire_level > 10:  # Example threshold for self-extinguishing
            aggregated_score *= 0.5  # Penalize task selection

        scores.append(aggregated_score)

    # Select the task with the highest score
    best_task_index = int(np.argmax(scores))
    
    return best_task_index