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
    # Initialize the temperature parameters for score normalization
    distance_temperature = 10.0  # Temperature factor for distance scaling
    intensity_temperature = 5.0  # Temperature factor for fire intensity scaling
    weight_temperature = 3.0     # Temperature factor for priority weight scaling

    # Calculate a score for each fire task
    best_task_index = -1
    best_task_score = float('-inf')
    
    for i in range(len(fire_pos)):
        # Distance-based score
        fire_y, fire_x = fire_pos[i]
        distance = ((agent_pos[0] - fire_y) ** 2 + (agent_pos[1] - fire_x) ** 2) ** 0.5
        distance_score = -distance / distance_temperature  # Penalize farther tasks

        # Fire intensity score
        fire_intensity_score = fire_intensities[i] / intensity_temperature  # Favor higher intensity fires

        # Priority weight score
        priority_score = fire_putout_weight[i] / weight_temperature  # Favor tasks with high weights

        # Agent capability score
        remaining_fire_intensity = max(0, fire_intensities[i] - (agent_fire_reduction_power * agent_suppressant_num))
        agent_effectiveness_score = -remaining_fire_intensity  # Prefer tasks where agent can have high suppression effect

        # Combine all scores
        total_score = (
            distance_score +
            fire_intensity_score +
            priority_score +
            agent_effectiveness_score
        )

        # Update the best task
        if total_score > best_task_score:
            best_task_score = total_score
            best_task_index = i

    return best_task_index