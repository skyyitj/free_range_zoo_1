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
    scores = []
    
    dist_temperature = 0.2
    level_temperature = 3.0
    intensity_temperature = 2.0
    weight_temperature = 0.5
    effect_temperature = 0.1
    
    for i in range(num_tasks):
        dist = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        effect = agent_suppressant_num * agent_fire_reduction_power / max(fire_intensities[i], 1)
        score = -np.exp(-dist/dist_temperature) \
                -np.exp(-fire_levels[i]/level_temperature) \
                -np.exp(-fire_intensities[i]/intensity_temperature) \
                +np.exp(fire_putout_weight[i]/weight_temperature) \
                +np.exp(effect / effect_temperature)
        scores.append(score)
        
    return np.argmax(scores)