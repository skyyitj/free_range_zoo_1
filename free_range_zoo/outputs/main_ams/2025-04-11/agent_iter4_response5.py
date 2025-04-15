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
    def distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    if agent_suppressant_num <= 0:
        return 0  # If the suppressant is empty, stay in the current position to recharge

    distance_to_fire = [distance(agent_pos, fire) for fire in fire_pos]
    intensity_distance_ratio = [lvl/dst for lvl, dst in zip(fire_intensities, distance_to_fire)]

    task_index = intensity_distance_ratio.index(max(intensity_distance_ratio))

    return task_index