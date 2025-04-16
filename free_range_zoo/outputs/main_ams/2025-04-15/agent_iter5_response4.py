from scipy.spatial.distance import cdist
import numpy as np

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
    intensity_temperature = 1.0
    weight_temperature = 0.5
    effect_temperature = 1.0
    agent_distribution_temperature = 0.3
    
    # Calculate the distribution of agents
    agent_distribution = [(cdist([agent_pos], fire_pos).argmin()) for agent_pos in other_agents_pos]
    
    for i in range(num_tasks):
        # Calculate Euclidean distance from agent position to fire position
        dist = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        
        # Effect of agent's fire suppression on a particular fire
        effect = agent_suppressant_num * agent_fire_reduction_power / 
                  max(fire_intensities[i], 1)
        
        # Calculate score for each fire
        score = -np.exp(-dist / dist_temperature) \
                -np.exp(-fire_levels[i] / level_temperature) \
                -np.exp(-fire_intensities[i] / intensity_temperature) \
                +np.exp(fire_putout_weight[i] / weight_temperature) \
                +np.exp(effect / effect_temperature) \
                -np.exp(agent_distribution.count(i) / agent_distribution_temperature)
                
        scores.append(score)

    # Return the fire with the highest score
    return np.argmax(scores)