def single_agent_policy(
     agent_pos: Tuple[float, float],
     agent_fire_reduction_power: float,
     agent_suppressant_num: float,
     other_agents_pos: List[Tuple[float, float]],
     fire_pos: List[Tuple[float, float]],
     fire_levels: List[float],
     fire_intensities: List[float],
 ) -> int:

    # Calculating the distance between the agent and fires
    distances = [np.sqrt((fire[0] - agent_pos[0]) ** 2 + (fire[1] - agent_pos[1]) ** 2) for fire in fire_pos]
    
    # Identifying fires within the agent's reach
    reachable_fires = [i for i, distance in enumerate(distances) if distance <= agent_suppressant_num]
    
    # No reachable fires, recharge the suppressant 
    if not reachable_fires:
        return -1

    # Calculate the efficiency of tackling each fire. This is balanced between the intensity of the fire and the amount of suppressant it needs
    fire_efficiencies = [fire_intensities[i] / fire_levels[i] for i in reachable_fires]

    # Identify the fire with the highest efficiency to tackle using the primary objective
    high_efficiency_fire = max(zip(reachable_fires, fire_efficiencies), key=lambda x: x[1])[0]

    return high_efficiency_fire