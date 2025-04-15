def single_agent_policy(
    agent_pos,                    # (y, x)
    agent_fire_reduction_power,  # float
    agent_suppressant_num,       # float
    other_agents_pos,            # List[(y, x)]
    fire_pos,                    # List[(y, x)]
    fire_levels,                 # List[float]
    fire_intensities             # List[float]
):