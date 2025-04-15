def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    # compute the distances to all fires
    distances = [np.sqrt((fire[0] - agent_pos[0])**2 + (fire[1] - agent_pos[1])**2) 
                 for fire in fire_pos]
    
    # list all fires that can be reached without recharging
    fires_within_reach = [i for i, dist in enumerate(distances) 
                          if dist <= agent_suppressant_num/agent_fire_reduction_power]
    
    if fires_within_reach:
        # if fires can be extinguished, prefer the ones with highest intensity
        fires_within_reach.sort(key=lambda i: -fire_intensities[i])
        
        # but also consider the level of the fire if intensities are close
        if len(fires_within_reach) > 1 and \
           fire_intensities[fires_within_reach[0]] - fire_intensities[fires_within_reach[1]] < 0.1:
            fires_within_reach.sort(key=lambda i: -fire_levels[i])
        
        # avoid fires that are already been targeted by other agents
        fires_within_reach = [i for i in fires_within_reach 
                              if not any(np.sqrt((fire_pos[i][0] - pos[0])**2 + 
                                                 (fire_pos[i][1] - pos[1])**2) 
                                         <= suppressant_num/fire_reduction_power 
                                         for pos, fire_reduction_power, suppressant_num in 
                                         zip(other_agents_pos, agent_fire_reduction_powers, agent_suppressant_nums))]
        
        if fires_within_reach:
            return fires_within_reach[0]
    
    # if no fires can be extinguished or all are already being targeted by other agents, recharge
    return -1