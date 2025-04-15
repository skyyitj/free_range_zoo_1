import numpy as np
from collections import Counter
import operator

def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:
    
    num_fires = len(fire_pos)

    if agent_suppressant_num <= 0:
        return -1

    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]
    agent_counts = Counter([np.argmin([np.sqrt((ag[0]-fire[0])**2 + (ag[1]-fire[1])**2) for fire in fire_pos])+1 for ag in other_agents_pos])
    
    # Assign high initial value for minimum score
    min_distance_score = np.inf
    min_distance_task = -1

    for task_id in range(num_fires):

        # This fire's suppressant efficiency score 
        efficiency_score = (fire_intensities[task_id]*agent_fire_reduction_power*agent_suppressant_num) / (1 + agent_counts.get(task_id, 0))

        # Calculate a distance score
        distance_score = distances[task_id] / efficiency_score
        
        # Update minimum distance score and task id
        if distance_score < min_distance_score:
            min_distance_score = distance_score
            min_distance_task = task_id
    
    return min_distance_task