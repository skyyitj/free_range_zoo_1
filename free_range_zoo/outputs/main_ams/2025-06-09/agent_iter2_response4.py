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
    distance_temp = 5.0  
    effectiveness_temp = 0.1  
    importance_temp = 0.5  

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        possible_suppressant_use = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        if possible_suppressant_use < 1:
            # If the agent lacks resources to make a significant impact, it will minimize resource use
            continue

        potential_effectiveness = agent_fire_reduction_power * possible_suppressant_use
        
        importance_weight = fire_putout_weight[task_index]

        # Scoring adjustment to balance the use of suppressants and the focus on important tasks
        task_score = (
            -np.log1p(distance / distance_temp) * 2.0 +  # Increase the impact of distance
            np.log1p(potential_effectiveness / effectiveness_temp) * 3.0 +  # Amplify effectiveness influence
            np.log1p(importance_weight / importance_temp) * 5.0  # Increased weight to task importance
        )

        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index