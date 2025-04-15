def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    
    from collections import Counter
    from operator import itemgetter
    
    agent_fires = []
    for i, fire_level in enumerate(fire_levels):
        if fire_level*agent_fire_reduction_power <= agent_suppressant_num:
            other_agents_fire = Counter([abs(other_agent-fire_pos[i]) for other_agent in other_agents_pos])
            closest_agent = min(other_agents_fire)
            fire_priority = fire_level*fire_intensities[i]/(1 + other_agents_fire[closest_agent])
            agent_fires.append((i, fire_priority))

    agent_fires.sort(key=itemgetter(1), reverse=True)
    
    return agent_fires[0][0] if len(agent_fires)>0 else None