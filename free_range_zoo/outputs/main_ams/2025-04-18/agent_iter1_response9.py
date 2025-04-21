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
):
    temp = 1.0  # The temperature parameter
    max_score = float('-inf')
    best_fire = None

    # Calculate total fire intensity for efficiency calculation
    total_fire_intensity = np.sum(fire_intensities)

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate agent's efficiency
        efficiency = agent_fire_reduction_power / total_fire_intensity
        
        # Calculate distance from agent to each fire
        dist = ((fire_position[0] - agent_pos[0])**2 + (fire_position[1] - agent_pos[1])**2)**0.5

        # Calculate score for each fire
        # Priority weights, fire intensity and agent efficiency increases the score, while distance decreases it
        # Score = (efficiency × Weight × Fire Intensity) / (Distance + temp)
        score = (np.log(efficiency) * fire_weight * np.log(fire_intensity)) / (np.log(dist) + temp)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire