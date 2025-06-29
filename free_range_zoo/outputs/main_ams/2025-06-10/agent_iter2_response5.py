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
    import numpy as np

    num_tasks = len(fire_pos)
    task_scores = []
    
    # Temperature parameters
    distance_temperature = 0.01
    remaining_fire_temperature = 1.0
    resource_temperature = 0.1
    
    # Calculate the score for each fire task
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        weight = fire_putout_weight[i]
        
        # Calculate the distance to the fire
        agent_y, agent_x = agent_pos
        distance = np.sqrt((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2)

        # Calculate remaining fire estimate
        potential_fire_suppression = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensity)
        remaining_fire = fire_intensity - potential_fire_suppression

        # Ensure no negative values for remaining fire
        remaining_fire = max(0, remaining_fire)

        # Calculate scores and apply weights and transformations
        distance_score = np.exp(-distance * distance_temperature)
        remaining_fire_score = np.exp(-remaining_fire * remaining_fire_temperature)
        resource_score = np.exp(-abs(agent_suppressant_num) * resource_temperature)
        
        total_score = weight * distance_score * remaining_fire_score * resource_score
        
        task_scores.append(total_score)
    
    # Choose the task with the highest total score
    selected_task_index = np.argmax(task_scores)
    return selected_task_index