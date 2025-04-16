from scipy.spatial.distance import cdist

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
    
    dist_temperature = 0.3
    level_temperature = 4.0
    intensity_temperature = 1.5
    weight_temperature = 0.4
    effect_temperature = 0.4
    agent_distribution_temperature = 0.3
    capacity_temperature = 0.5
    
    # Calculate number of agents going to each fire
    agent_distribution = [(cdist([agent_pos], fire_pos).argmin()) for agent_pos in other_agents_pos]

    for i in range(num_tasks):
        dist = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        effect = agent_suppressant_num * agent_fire_reduction_power / max(fire_intensities[i], 1)
        tasks_capacity = agent_suppressant_num / max(fire_levels[i], 1)
        score = -np.exp(-dist/dist_temperature) \
                -np.exp(-fire_levels[i]/level_temperature) \
                -np.exp(-fire_intensities[i]/intensity_temperature) \
                +np.exp(fire_putout_weight[i]/weight_temperature) \
                +np.exp(effect/effect_temperature) \
                -np.exp(agent_distribution.count(i)/agent_distribution_temperature) \
                +np.exp(tasks_capacity/capacity_temperature)

        scores.append(score)

    return np.argmax(scores)