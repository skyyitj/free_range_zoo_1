import numpy as np

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

    max_score = float('-inf')
    best_fire = None

    for i in range(len(fire_pos)):
        # Distance
        dist = np.sqrt((fire_pos[i][0]-agent_pos[0])**2 + (fire_pos[i][1]-agent_pos[1])**2)
        
        # Scoring based on suppression power, intensity and weight of fire and adjusted for distance
        score = (agent_fire_reduction_power * min(agent_suppressant_num, fire_intensities[i]) * fire_putout_weight[i]) / (1 + dist)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire