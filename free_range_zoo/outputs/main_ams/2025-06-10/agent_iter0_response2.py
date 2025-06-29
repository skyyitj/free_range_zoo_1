from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_supressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    import numpy as np

    num_fires = len(fire_pos)
    
    # Temperature constants for transformations
    distance_temp = 1.0
    intensity_temp = 1.0
    resource_temp = 1.0
    
    def distance(y1, x1, y2, x2):
        return np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
    
    best_task_index = -1
    highest_score = -float('inf')
    
    for i in range(num_fires):
        fire_location = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_resource_weight = fire_levels[i]
        fire_priority_weight = fire_putout_weight[i]
        
        # Calculate distance from agent position to the fire
        dist = distance(agent_pos[0], agent_pos[1], fire_location[0], fire_location[1])
        distance_score = np.exp(-dist / distance_temp)  # Closer fires should have higher score
        
        # Intensity score (high intensity should have a higher score if we have resources)
        if agent_supressant_num * agent_fire_reduction_power / fire_intensity > 1:
            intensity_score = np.exp(fire_intensity / intensity_temp)
        else:
            intensity_score = np.exp(-fire_intensity / intensity_temp)
        
        # Resource conservation score (prefer tasks where we use less resources if resources are low)
        resource_usage = min(agent_supressant_num, fire_intensity / agent_fire_reduction_power)
        resource_score = np.exp((-resource_usage / agent_supressant_num) / resource_temp)
        
        # Prioritization score (based on weights)
        priority_score = fire_priority_weight  # Higher weight should have higher score
        
        # Composite score for deciding on the task to pick
        task_score = (distance_score + intensity_score + resource_score) * priority_score
        
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = i
            
    return best_task_index