def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    fire_putout_weight: List[float]
) -> int:

    # Initialize a negative infinity maximum score and a null best fire
    highest_score = float('-inf')
    selected_fire = None

    # Iterate through each fire to calculate the task priority score
    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance between the agent and the fire weighted by available agent's suppressant
        distance = (((fire_position[0] - agent_pos[0]) ** 2 + (fire_position[1] - agent_pos[1]) ** 2) ** 0.5) / agent_suppressant_num

        # Considering the agent's fire suppression capability, available suppressant, and the fire intensity.
        suppressant_factor = (agent_fire_reduction_power / fire_intensity) * agent_suppressant_num

        # Consider the fire weight, suppressant factor, and distance to calculate the fire score.
        score = fire_weight * suppressant_factor - distance * fire_intensity

        # if the calculated score is higher than the current highest score, update the highest score and best fire
        if score > highest_score:
            highest_score = score
            selected_fire = i

    # Return the index of the selected fire.
    return selected_fire