from typing import List, Tuple

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

    num_fires = len(fire_pos)
    num_agents = 1 + len(other_agents_pos)
    
    # Calculate distances from each agent to each fire
    dist = [[((y-x[0])**2 + (z-x[1])**2)**0.5 for y, z in fire_pos] for x in [agent_pos, *other_agents_pos]]
    
    # Calculate the minimum distance to each fire for all agents
    min_dist = [min(x) for x in zip(*dist)]
    
    # Calculate fire priorities as the weighted average of fire level and fire intensity, adjusted by the fire weight
    fire_priority = [a*b*c/(d**2+1) for a, b, c, d in zip(fire_levels, fire_intensities, fire_putout_weight, min_dist)]
    
    # If another agent is closer to a fire, decrease its priority
    for i in range(num_fires):
        if min_dist[i] < dist[0][i]:
            fire_priority[i] *= 0.5
    
    # Return the fire with the highest priority
    return fire_priority.index(max(fire_priority))