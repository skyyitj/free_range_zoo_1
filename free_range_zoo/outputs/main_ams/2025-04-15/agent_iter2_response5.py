python
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

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Increase level_temperature, reduce intensity_temperature
    level_temperature = 0.3
    intensity_temperature = 0.05
    distance_temperature = 0.01

    for task in range(num_tasks):

        fire_distance = distance.euclidean(agent_pos, fire_pos[task])

        scores[task] = (
            np.exp(-fire_levels[task] * level_temperature) +
            np.exp(-fire_intensities[task] / can_put_out_fire * intensity_temperature)
            if can_put_out_fire >= fire_intensities[task] else 0
            - np.exp(fire_distance * distance_temperature)
        ) * fire_putout_weight[task]
    
    max_score_task = np.argmax(scores)
    return max_score_task