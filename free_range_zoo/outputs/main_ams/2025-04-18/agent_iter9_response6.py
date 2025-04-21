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

    # Observation: The agent is using too much suppressant ("Average Used Suppressant" is high),
    # and it doesn't seem to be effectively extinguishing the fires ("Average Putout Number" is low)
    # Also, the agent does not seem effective in controlling the fires ("Average Fire Intensity Change" is low).

    # This means an agent might be using too much suppressant ineffectively, and
    # blindly trying to put out fires even though it doesn't have enough fire reduction power.
    # The agent should consider its fire reduction power more to make sure it doesn't waste suppressant.

    # Thus, we adjust the temperatures to balance the metric components. We increase the temperature value for suppressant
    # to create an effect of preserving it and decrease the fire level temperature to account for fire levels more.

    max_score = -float('inf')
    best_fire = None

    dist_temperature = 0.2  # Adjusted to put higher weight on distance.
    suppress_power_temperature = 0.2  # Increase the value to account for suppressing power more.
    fire_level_temperature = 0.5  # Decreased to put less importance on the fire level.

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
  
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5 / (agent_suppressant_num+1e-7)
        suppression_power = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity+1)

        score = np.exp((fire_weight * (fire_level+1e-7) / (dist_temperature * dist + 1) + suppression_power * suppress_power_temperature) / fire_level_temperature)

        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire