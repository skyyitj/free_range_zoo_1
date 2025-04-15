import numpy as np
from typing import List, Tuple

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    # Process if agent should recharge its suppressant
    if agent_suppressant_num <= 0:
        return -1

    fire_distances = [np.sqrt(np.sum((np.array(agent_pos) - np.array(pos))**2)) for pos in fire_pos]
    
    # Filter reachable fires and calculate the efforts required
    reachable_fires = [i for i, dist in enumerate(fire_distances) if dist<=agent_suppressant_num]
    if not reachable_fires: return -1  
    
    efforts = [fire_intensities[i]*fire_levels[i]/agent_fire_reduction_power for i in reachable_fires]
    
    # Choose the fire that requires the least efforts
    return reachable_fires[np.argmin(efforts)]