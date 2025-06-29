def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_supressant_num: float, 
    other_agents_pos: List[Tuple[float, float]], 
    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int], 
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')

    # Temperature constants
    distance_temp = 0.1  # Higher sensitivity to distance, prefer tasks closer
    effectiveness_temp = 5.0  # Strong emphasis on potential effectiveness
    efficiency_temp = 2.0  # Reward efficiency of suppressant usage
    
    # Iterate tasks and calculate scores
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Euclidean distance to fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)

        # Max effective suppressant this agent can use
        max_effective_suppressant = min(agent_fire_reduction_power * agent_supressant_num, fire_intensity)

        # Potential intensity reduction
        potential_reduction = min(max_effective_suppressant, fire_intensity)

        # Suppressant efficiency: intensity reduction per unit suppressant
        if potential_reduction > 0:
            suppressant_efficiency = potential_reduction / (potential_reduction / agent_fire_reduction_power)
        else:
            suppressant_efficiency = 0

        # Importance of the task based on predefined weight
        importance_weight = fire_putout_weight[task_index]
        
        # Calculate score considering distance, effectiveness, and efficiency
        task_score = (
            -np.exp(distance * distance_temp) +  # Penalize distance more heavily
            np.log1p(potential_reduction) * effectiveness_temp +  # Strongly weight effective potential reduction
            np.log1p(suppressant_efficiency) * efficiency_temp +  # Encourage efficient use of suppressants
            importance_weight  # Include the importance weight as is
        )

        # Update the best task based on score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index