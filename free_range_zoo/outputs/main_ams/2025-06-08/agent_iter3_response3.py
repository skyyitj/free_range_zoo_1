def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              
    agent_fire_reduction_power: float,           
    agent_suppressant_num: float,                

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], 

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         
    fire_levels: List[int],                      
    fire_intensities: List[float],               

    # === Task Prioritization ===
    fire_putout_weight: List[float],             
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    import math

    # Temperature parameters for policy tuning
    reward_temperature = 3.5  # Emphasize high-reward fires
    intensity_temperature = 4  # Prioritize high-intensity fires
    suppressant_temperature = 2.5  # Balance suppressant efficiency
    burnout_temperature = 3.5  # Penalize fires at risk of burning out
    distance_temperature = 9  # Slight focus on proximity

    scores = []
    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos

        # --- Reward Component ---
        reward_score = math.exp(fire_putout_weight[i] / reward_temperature)

        # --- Fire Intensity Component ---
        intensity_score = math.exp(fire_intensities[i] / intensity_temperature)

        # --- Resource Efficiency Component ---
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        suppressant_score = (
            math.exp(-required_suppressant / suppressant_temperature)
            if agent_suppressant_num >= required_suppressant else 0
        )

        # --- Burnout Risk Component ---
        burnout_penalty = math.exp(fire_levels[i] / burnout_temperature)

        # --- Distance Component ---
        distance = math.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = math.exp(-distance / distance_temperature)

        # --- Combine Scores ---
        total_score = (
            reward_score *
            intensity_score *
            suppressant_score *
            distance_score /
            (burnout_penalty + 1e-6)  # Prevent division by zero
        )
        scores.append(total_score)

    # Choose the task with the highest score
    return int(scores.index(max(scores)))