import numpy as np

def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
) -> int:
    num_tasks = len(fire_pos)
    
    # Scoring components temperatures
    distance_temp = 0.1
    intensity_temp = 0.05
    weight_temp = 1.0
    suppressant_temp = 0.1
    
    best_task = -1
    best_score = float('-inf')
    
    # Iterate over each fire task to compute scores based on fires
    for i in range(num_tasks):
        fy, fx = fire_pos[i]
        intensity = fire_intensities[i]
        level = fire_levels[i]
        weight = fire_putout_weight[i]
        
        # Euclidean distance from agent to the fire
        dist = np.sqrt((fy - agent_pos[0]) ** 2 + (fx - agent_pos[1]) ** 2)
        
        # Consider agent effectiveness - how much of the fire they could potentially suppress
        potential_suppression = min(agent_fire_reduction_power * agent_suppressant_num, level)
        fire_after_suppression = level - potential_suppression
        
        # Scoring
        distance_score = np.exp(-dist * distance_temp)
        suppression_score = np.exp(-fire_after_suppression * intensity_temp)
        weight_score = np.exp(weight * weight_temp)
        suppressant_score = np.exp(-potential_suppression * suppressant_temp)
        
        score = (
            distance_score * 
            suppression_score * 
            weight_score * 
            suppressant_score
        )
        
        if score > best_score:
            best_score = score
            best_task = i
    
    return best_task