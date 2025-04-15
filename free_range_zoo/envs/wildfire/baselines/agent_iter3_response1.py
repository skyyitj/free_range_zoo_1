def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    suppressant_needed = [level/intensity for level, intensity in zip(fire_levels, fire_intensities)]
    task_indices = [i for i, s in enumerate(suppressant_needed) if s <= agent_suppressant_num]

    # If no fire is small enough to be fought by the agent, return -1 signalling 'no operation'
    if not task_indices:
        return -1

    largest_fire_index = max(task_indices, key=fire_levels.__getitem__)
    return largest_fire_index