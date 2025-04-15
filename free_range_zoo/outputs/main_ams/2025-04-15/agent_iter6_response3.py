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

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = 1 / (1 + np.exp(-agent_suppressant_num * agent_fire_reduction_power))

    # Adjusted temperature values
    level_temperature = 0.5
    intensity_temperature = 0.25
    distance_temperature = 0.05  # Decreased sensitivity to distance

    for task in range(num_tasks):
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        scores[task] = (
            fire_putout_weight[task] * np.sqrt(agent_suppressant_num) +  # Added bonus  
            np.exp(-fire_levels[task] * level_temperature) +
            can_put_out_fire * np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature) -
            fire_distance * np.exp(fire_distance * distance_temperature)
        ) 

    max_score_task = np.argmax(scores)
    return max_score_task