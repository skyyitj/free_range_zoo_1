import math

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    agent_y, agent_x = agent_pos

    # Do nothing when no suppressants are left.
    if agent_suppressant_num == 0:
        return -1

    assert len(fire_pos) == len(fire_levels) == len(fire_intensities)

    # Calculate agent priority by using agent's distance to the fire and current fire levels.
    fire_priorities = []
    for fire_position, fire_level, fire_intensity in zip(fire_pos, fire_levels, fire_intensities):
        fire_y, fire_x = fire_position
        distance = math.sqrt((fire_x - agent_x) ** 2 + (fire_y - agent_y) ** 2)
        
        # The lower the fire level, the higher the priority.
        # The greater the fire intensity, the higher the priority.
        # The closer the distance, the higher the priority.
        fire_priority = ((max(fire_levels) - fire_level + 1) * fire_intensity) / distance
        fire_priorities.append(fire_priority)

    # Choose the fire with the highest priority.
    chosen_fire = fire_priorities.index(max(fire_priorities))

    # Check if any other agent is closer to the chosen fire.
    for other_agent_pos in other_agents_pos:
        other_agent_y, other_agent_x = other_agent_pos
        other_agent_distance = math.sqrt((fire_x - other_agent_x) ** 2 + (fire_y - other_agent_y) ** 2)

        # If another agent is closer, this agent does nothing.
        if other_agent_distance < distance:
            return -1

    # Otherwise, the agent goes to extinguish the chosen fire.
    return chosen_fire