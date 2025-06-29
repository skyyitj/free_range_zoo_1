def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_suppressant_num: float, 
    other_agents_pos: List[Tuple[float, float]], 
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')

    # Adjust temperature for distance evaluation
    distance_temp = 2.0  # Lower the impact of distance slightly
    effectiveness_temp = 0.5  # Adjust to improve suppressant efficiency
    importance_temp = 0.3  # Slightly reduce the impact of importance scale

    available_resources = agent_suppressant_num
    suppressant_threshold = 0.1 * available_resources  # Define a threshold to manage suppressant

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        if distance > available_resources:
            # If distance is greater than available resources, the agent is unlikely to be effective
            continue

        possible_suppressant_use = min(available_resources, fire_intensity / agent_fire_reduction_power)
        if possible_suppressant_use < suppressant_threshold:
            # If the agent can't use enough suppressant to be effective, continue
            continue

        potential_effectiveness = available_resources * max(1, agent_fire_reduction_power * fire_level)

        importance_weight = fire_putout_weight[task_index]

        # Refine scoring function
        task_score = (
            -np.log1p(distance / distance_temp) * 1.5 +
            np.log1p(potential_effectiveness / effectiveness_temp) * 4.0 +
            np.log1p(importance_weight / importance_temp) * 4.5
        )

        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index