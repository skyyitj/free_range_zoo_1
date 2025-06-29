def single_agent_policy(agent_pos: Tuple[float, float], 
                        agent_fire_reduction_power: float, 
                        agent_supressant_num: float, 
                        other_agents_pos: List[Tuple[float, float]], 
                        fire_pos: List[Tuple[float, float]], 
                        fire_levels: List[int], 
                        fire_intensities: List[float],
                        fire_putout_weight: List[float]) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')
    
    distance_temp = 0.2  # Further decrease so closer fires are focused upon more
    effectiveness_temp = 4.0  # Enhanced focus on effectiveness
    importance_temp = 2.5  # Tuning task importance slightly
    suppressant_eff_temp = 3.5  # More focus on efficient usage of suppressants

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate distance to the fire
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        # Optimal suppressant use estimation
        max_effective_suppressant = min(fire_intensity / agent_fire_reduction_power if agent_fire_reduction_power > 0 else float('inf'), agent_supressant_num)
        
        # Effective reduction in fire intensity achievable by the agent
        potential_effectiveness = agent_fire_reduction_power * max_effective_suppressant
        
        # Efficiency of suppressant usage
        suppressant_efficiency = potential_effectiveness / max_effective_suppressant if max_effective_suppressant > 0 else 0
        
        importance_weight = fire_putout_weight[task_index]
        
        # Compose the task scoring metric
        task_score = (
            -np.exp(distance / distance_temp) +
            np.log(potential_effectiveness + 1) * effectiveness_temp +
            np.log(suppressant_efficiency + 1) * suppressant_eff_temp +
            np.exp(importance_weight) * importance_temp
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index