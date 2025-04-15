def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_agents = len(other_agents_pos) + 1
    num_fires = len(fire_pos)
    scores = []
    for i in range(num_fires):
        # Estimate time to reach the fire
        distance = sum(abs(a-b) for a, b in zip(agent_pos, fire_pos[i]))
        time_to_reach = distance 

        # Estimate fire suppressant use to control fire
        suppressant_use = min(agent_suppressant_num, fire_intensities[i] / agent_fire_reduction_power)

        # Estimate time to control fire, based on estimated suppressant use
        time_to_control = suppressant_use / agent_fire_reduction_power

        # Compute score as ratio of reward weight to effort
        fire_score = fire_putout_weight[i] / (time_to_reach + time_to_control)

        # Append score to list
        scores.append(fire_score)

    # Return index of fire with highest score
    return scores.index(max(scores))