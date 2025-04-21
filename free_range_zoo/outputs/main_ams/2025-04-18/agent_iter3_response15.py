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

    # weight factors
    weight_dist = 0.2
    weight_intensity = 0.4
    weight_level = 0.2
    weight_resource = 0.2

    best_fire = None
    best_score = -np.inf

    for i, (pos, level, intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):

        # position factors
        self_dist = np.sqrt((pos[0] - agent_pos[0])**2 + (pos[1] - agent_pos[1])**2)
        other_dist = [np.sqrt((pos[0] - o_pos[0])**2 + (pos[1] - o_pos[1])**2) for o_pos in other_agents_pos]
        
        min_other_dist = min(other_dist) if other_dist else np.inf
        dist_factor = self_dist / (1 + min_other_dist)

        # fire factors
        level_factor = level
        intensity_factor = intensity

        # agent factors
        resource_factor = agent_suppressant_num / (1 + intensity)

        score = (fire_putout_weight[i] 
                 - weight_dist * dist_factor 
                 - weight_level * level_factor 
                 - weight_intensity * intensity_factor 
                 - weight_resource * resource_factor)

        if score > best_score:
            best_score = score
            best_fire = i

    return best_fire