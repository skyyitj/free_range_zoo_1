def calculate_moves(agent_pos: Tuple[float, float], task_pos: Tuple[float, float]) -> float:
    return ((task_pos[0] - agent_pos[0]) ** 2 + (task_pos[1] - agent_pos[1]) ** 2) ** 0.5

def calculate_task_power(fire_level: float, fire_intensity: float) -> float:
    return fire_level * fire_intensity

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
    valid_action_space: List[List[int]]
) -> int:
    
    # Initialize maximum task power and chosen task index
    max_task_power = -1
    chosen_task_index = -1
    
    # Loop through each fire task
    for task_index, (task_pos, fire_level, fire_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        
        # calculate the distance from agent to the task
        moves_to_task = calculate_moves(agent_pos, task_pos)
        
        # check if agent has enough suppressant to complete the task
        if moves_to_task + 1 > agent_suppressant_num:
            continue
            
        # calculate the task power of the fire task based on its level and intensity
        task_power = calculate_task_power(fire_level, fire_intensity)
        
        # Add up the distances of all other agents to the task for collaborative firefighting
        for other_agent_pos in other_agents_pos:
            moves_to_task += calculate_moves(other_agent_pos, task_pos)
            
        # Balance the evaluation with a division to avoid overly patrolling in low intensity fire. The lower the value, the better
        evaluation = moves_to_task / task_power
        
        # check if this task's evaluation is the lowest
        if evaluation < max_task_power or max_task_power == -1:
            max_task_power = evaluation
            chosen_task_index = task_index
    
    return chosen_task_index
```