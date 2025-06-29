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

    # Adjustments on temperature parameters
    distance_temp = 2.0
    effectiveness_temp = 1.5  # Encouraging suppression effectiveness
    importance_temp = 1.0  # Weighting the importance
    
    urgency_temp = 2.0  # To prioritize urgent fires which are burning fiercely or close to important assets
    
    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculating Euclidean distance to the fire
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
       
        # Evaluating suppressant usage, focusing on not running out immediately
        possible_suppressant_use = min(agent_supressant_num, fire_intensity / agent_fire_reduction_power)
        
        # Evaluate potential effectiveness
        potential_effectiveness = agent_fire_reduction_power * possible_suppressant_use
        
        urgency = fire_intensity / (distance + 1)  # Fires with high intensity and closer should be more urgent
        
        importance_weight = fire_putout_weight[task_index]
       
        # Recalculated task score incorporating urgency and improved effectiveness consideration
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(potential_effectiveness + 1) * effectiveness_temp +
            importance_weight * importance_temp + 
            np.log(urgency + 1) * urgency_temp
        )
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index