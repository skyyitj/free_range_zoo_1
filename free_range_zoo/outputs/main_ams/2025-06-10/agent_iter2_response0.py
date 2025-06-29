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

    # Parameters for score transformations
    distance_temp = 0.1
    intensity_temp = 0.05
    reward_temp = 5.0
    
    best_task = -1
    highest_score = -np.inf
    
    for index in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[index][0]) ** 2 + (agent_pos[1] - fire_pos[index][1]) ** 2)
        weighted_distance = np.exp(-distance_temp * distance)
        
        fire_intensity = fire_intensities[index]
        intensity_score = np.exp(-intensity_temp * fire_intensity)
        
        reward_weight = fire_putout_weight[index]
        reward_score = np.exp(reward_temp * reward_weight)
        
        # Combining scores with a weighting that reflects their importance
        total_score = weighted_distance * intensity_score * reward_score
        
        # Select the fire task with the highest score
        if total_score > highest_score:
            highest_score = total_score
            best_task = index
            
    return best_task