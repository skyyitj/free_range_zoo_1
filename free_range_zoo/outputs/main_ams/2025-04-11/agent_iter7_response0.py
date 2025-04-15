import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num <= 0:
        # recharge if out of suppressant
        return -1

    # calculate distance to each fire
    distances_to_fires = np.sqrt(np.sum((np.array(fire_pos) - np.array(agent_pos))**2, axis=1))
    reachable_fires = np.where(distances_to_fires<= agent_suppressant_num)[0]

    if reachable_fires.size == 0:
        # no fires can be extinguished with the current amount of suppressant, it's better to recharge
        return -1 

    # Calculate the estimated number of suppressant applications required to extinguish each fire
    suppressants_needed = np.array(fire_levels) / agent_fire_reduction_power

    # For reachable fires, get the fire that requires the least suppressants to extinguish
    chosen_fire = reachable_fires[np.argmin(suppressants_needed[reachable_fires])]
    
    return chosen_fire