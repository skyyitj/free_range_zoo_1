def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]], 

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[int],                    
    fire_intensities: List[float], 

    fire_putout_weight: List[float], 
) -> int:

    num_tasks = len(fire_pos)  
    scores = []                                  
    
    # Temperature parameters for the exponential functions
    dist_temperature = 5.0
    level_temperature = 7.0
    intensity_temperature = 0.50 
    suppressant_temperature = 0.50
    weight_temperature = 0.2
  
    # Iterate over tasks to assign scores
    for i in range(num_tasks):
        # calculate distance to fire
        agent2fire_dist = ((agent_pos[0] - fire_pos[i][0])**2 + 
                      (agent_pos[1] - fire_pos[i][1])**2)**0.5

        # calculate agent's possible contribution to reducing the fire
        # considering the suppressant quantity and agent's ability
        agent_effect = agent_suppressant_num * agent_fire_reduction_power 

        # calculate how much a fire can be possibly reduced 
        reduced_fire_intensity = max(fire_intensities[i] - agent_effect, 0)

        # Score based on distance, fire level, fire intensity and task weight
        score = -np.exp(-agent2fire_dist / dist_temperature) \
                -np.exp(-fire_levels[i] / level_temperature) \
                -np.exp(-reduced_fire_intensity / intensity_temperature) \
                +np.exp(agent_suppressant_num / suppressant_temperature) \
                +np.exp(fire_putout_weight[i] / weight_temperature)
        scores.append(score)

        
    return np.argmax(scores)