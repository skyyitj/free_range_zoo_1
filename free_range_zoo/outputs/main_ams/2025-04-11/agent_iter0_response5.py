def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    if agent_suppressant_num < 1:
        return -1  # This agent needs to recharge; cannot action on a fire.

    num_tasks = len(fire_pos)
    agent_utility = [0] * num_tasks  # Utility to fight each fire

    for i in range(num_tasks):
        # Check if the fire is in extinguishable state, if not, skip calculation for this fire.
        if fire_levels[i] <= agent_fire_reduction_power and fire_intensities[i] < 3:  
            distance_to_fire = ((agent_pos[0]-fire_pos[i][0])**2 + (agent_pos[1]-fire_pos[i][1])**2) ** 0.5
            utility_without_cost = fire_levels[i] / distance_to_fire  # the higher the level and closer the fire, the higher the utility
    
            # reducing the utility if other agents are already heading to this fire position
            num_agents_heading_this_fire = other_agents_pos.count(fire_pos[i])

            if num_agents_heading_this_fire > 0:
                utility_without_cost /= (1 + num_agents_heading_this_fire)

            # ensure the suppressant is enough to put out the fire
            if fire_levels[i] <= agent_suppressant_num:
                agent_utility[i] = utility_without_cost

    # choosing the fire with the highest utility
    return agent_utility.index(max(agent_utility)) if max(agent_utility) > 0 else -1