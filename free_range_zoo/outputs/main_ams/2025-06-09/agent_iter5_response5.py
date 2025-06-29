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

    # Readjust temperature parameters
    distance_temp = 1.0  # Reassess the impact of distance
    effectiveness_temp = 2.0  # More attention on the capability to reduce the fire
    importance_temp = 3.0  # Further emphasize task importance given the performance influence
    
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Target usage considering intensities and level contribution, for better resource efficiency
        target_suppressant_use = min(agent_suppressant_num, fire_intensities[task_index] / agent_fire_reduction_power * 0.8)
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
      
        # Use task weight for importance scaling
        importance_weight = fire_putout_weight[task_index]
        
        # Score combining components taking into account newly adjusted temperatures
        task_score = (-np.exp(-(distance + 1)) / distance_temp +  # Reduced the distance attenuation
                      np.log(potential_effectiveness + 1) * effectiveness_temp +  # Increased focus on effectiveness
                      importance_weight * 10.0 / importance_temp  # Adjusted importance weight effects
                     )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index