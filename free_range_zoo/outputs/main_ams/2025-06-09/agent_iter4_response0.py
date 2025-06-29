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

    # Adjusting temperature parameters to attempt improvement in policy performance:
    distance_temp = 0.5   # Decrease sensitivity towards the distance to encourage addressing working on more strategic fires.
    effectiveness_temp = 1.5  # Increase effectiveness impact, encouraging actual fire reduction potential.
    importance_temp = 3.0  # Increasing weightage based on task priority to align more with important tasks.
    resource_use_temp = 0.8  # A new parameter to manage suppressant resource use carefully.

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate the Euclidean distance to each fire task
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        # Calculate possible effective suppressant use
        possible_suppressant_use = min(agent_supressant_num, fire_intensity / agent_fire_reduction_power)
        
        # If possible suppressant use is too low, it's not worth the effort, skip
        if possible_supressant_use < 1:
            continue
        
        # Calculate effectiveness based on possible suppressant use
        potential_effectiveness = agent_fire_reduction_power * possible_suppressant_use
        
        # Get importance weight from predefined fire weights
        importance_weight = fire_putout_weight[task_index]

        # Calculate the task score using the given heuristics and modified temperatures
        task_score = (
            -np.log(distance + 1) / distance_temp +  # Distance consideration
            np.log(potential_effectiveness + 1) * importance_temp +  # Effectiveness of fighting fire
            np.log(importance_weight + 1) * 5.0  # Importance of the task 
        )
        
        # Adjust score based on remaining suppressant, promoting conservative use
        task_score *= (agent_supressant_num / possible_suppressant_use) ** resource_use_temp
        
        # Update the best task based on the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index