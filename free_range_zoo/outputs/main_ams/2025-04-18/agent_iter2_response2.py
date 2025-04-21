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
    # === Fire Task Selection Scoring ===
    max_score = float('-inf')
    best_fire = None

    # Assign a higher weighting to distance to reduce steps
    distance_weight = 0.6

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / agent_suppressant_num
        dist *= distance_weight

        # Assign a higher score to fires with high intensities
        fire_score = np.exp(fire_intensity)
        
        suppressant_factor = (agent_fire_reduction_power / fire_intensity) * agent_suppressant_num
        score = fire_weight * suppressant_factor * fire_score - dist * fire_intensity  
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire