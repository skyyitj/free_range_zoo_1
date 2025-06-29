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

    # Adjust temperature constants for better balance based on metric performance feedback
    distance_temp = 0.15
    effectiveness_temp = 5.0
    importance_temp = 1.5
    suppressant_efficiency_temp = 1.0
    resource_conservation_temp = 0.8   # This new temperate emphasizes not burning through suppressant too quickly

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]

        # Calculate Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Strategy to calculate maximum effective suppressant use
        target_suppressant_use = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_suppressant_num)
        
        # Effective reduction in fire intensity achievable by this agent
        potential_effectiveness = agent_fire_reduction_power * target_suppressant_use
        
        # Calculate remaining suppressant after this action
        remaining_suppressant = agent_suppressant_num - target_suppressant_use

        suppressant_efficiency = potential_effectiveness / target_suppressant_use if target_suppressant_use > 0 else 0
        
        importance_weight = fire_putout_weight[task_index]
        
        # Calculate the task score considering task weight, distance, potential fire reduction, and suppressant efficiency
        task_score = (
            -np.exp(distance) / distance_temp +  # Empathize avoiding long-distanced tasks
            np.log(potential_effectiveness + 1) * effectiveness_temp +  # More importance to effectiveness
            np.log(suppressant_efficiency + 1) * suppressant_efficiency_temp +  # Rewarding suppressant efficiency
            np.exp(importance_weight) * importance_temp +  # Exp on importance weight, balance using importance temp
            np.log(remaining_suppressant + 1) * resource_conservation_temp  # Encouraging preservation of suppressant
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index