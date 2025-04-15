from typing import List, Tuple
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int: 

    if len(fire_pos) == 0:
        return -1  # return -1 to recharge if there are no fires

    if agent_suppressant_num <= 0:
        # recharge if out of suppressant
        return -1

    # Calculate distance to each fire
    distances_to_fire = np.sqrt(np.sum((np.array(fire_pos) - np.array([agent_pos]))**2, axis=1))
    
    # Only consider reachable fires
    reachable_fires = np.where(distances_to_fire <= agent_suppressant_num)[0]
    
    if len(reachable_fires) == 0:
        # no fires can be extinguished with the current amount of suppressant, it's better to recharge
        return -1  # return -1 to indicate that the agent should recharge
    
    # For each fire, calculate the estimated number of suppressant applications required to extinguish it
    suppressant_needed = fire_levels[reachable_fires] / agent_fire_reduction_power
    
    # Prioritize fires that need the least suppressant, but scale by inverse of fire intensity 
    # (we want to prioritize higher intensity fires IF we have the suppressant to deal with them)
    fire_priorities = suppressant_needed / np.array(fire_intensities)[reachable_fires]
    
    # Sort the fires by their priority, and select the one that needs the least suppressant to deal with
    chosen_fire = reachable_fires[np.argmin(fire_priorities)]
    return chosen_fire