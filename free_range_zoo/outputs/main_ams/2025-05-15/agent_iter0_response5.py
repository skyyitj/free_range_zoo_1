import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                    
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:

    num_fires = len(fire_intensities)
    decision_scores = []
    temp_dist = 0.1   # Temperature parameter for distance
    temp_intensity = 0.05  # Temperature parameter for fire intensity
    
    for i in range(num_fires):
        # Calculate the distance from the agent to the fire
        dy = agent_pos[0] - fire_pos[i][0]
        dx = agent_pos[1] - fire_pos[i][1]
        distance_to_fire = np.sqrt(dy**2 + dx**2)
        
        # Calculate the weighted score for the task:
        # use negative distance to prioritize closer fires 
        # multiply by the fire_putout_weight to prioritize weighted tasks
        score_distance = np.exp(-distance_to_fire/temp_dist)
        score_intensity = np.exp(-fire_intensities[i]/temp_intensity)
        
        # Combine scores with weights to form the final task score
        combined_score = score_distance * score_intensity * fire_putout_weight[i]
        
        decision_scores.append(combined_score)
    
    # Choose the fire with the highest score
    task_index = np.argmax(decision_scores)
    
    return task_index