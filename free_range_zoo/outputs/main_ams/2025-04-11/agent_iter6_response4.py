def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    distances_to_fires = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    fires_within_reach = [i for i, dist in enumerate(distances_to_fires) if dist <= agent_suppressant_num*agent_fire_reduction_power]

    if not fires_within_reach:
        # No reachable fire with the current suppressant, so recharge
        return -1

    fire_risks = [fire_levels[i]*fire_intensities[i] for i in fires_within_reach] #Fire risk is calculated as the multiplication of fire_level and fire_intensity
    fire_agents_nearby = [len([1 for agent in other_agents_pos if np.sqrt((fire_pos[i][0] - agent[0]) ** 2 + (fire_pos[i][1] - agent[1]) ** 2) <= agent_suppressant_num*agent_fire_reduction_power ]) for i in fires_within_reach] 
    
    # If some fires are already targeted by other agents,  decrease its value 
    fire_values = [risk/(1+agents_nearby) for risk, agents_nearby in zip(fire_risks, fire_agents_nearby)]
  
    # Choose the fire with the most value
    best_fire = max(range(len(fire_values)), key=fire_values.__getitem__)
    return fires_within_reach[best_fire]  # Return the original index