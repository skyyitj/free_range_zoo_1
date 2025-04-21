import math
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

    # === Scoring Fire Tasks ===
    max_score = float('-inf')
    best_fire = -1

    # compute pairwise distance to help avoid collaboration on the same fire
    pairwise_distance = np.zeros(len(fire_pos))

    for agent in other_agents_pos:
        dist = [((f[0]-agent[0])**2 + (f[1]-agent[1])**2)**0.5 for f in fire_pos]
        pairwise_distance = [max(d, x) for d, x in zip(dist, pairwise_distance)]
    
    for i in range(len(fire_pos)):
        # Distance to fire
        dist = ((fire_pos[i][0] - agent_pos[0])**2 + (fire_pos[i][1] - agent_pos[1])**2)**0.5

        # Firefighting efficiency
        firefighting_efficiency = agent_fire_reduction_power / (fire_intensities[i] + 1)
        
        # Score computation
        score = fire_putout_weight[i] * firefighting_efficiency / ((dist/agent_fire_reduction_power) + 1)
        
        # Prevent assigning the same fire task to multiple agents
        score /= pairwise_distance[i] + 1

        # Choose the task with the maximum score
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire