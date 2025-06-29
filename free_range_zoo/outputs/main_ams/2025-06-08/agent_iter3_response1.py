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
    distance_temperature = 7  # Moderate focus on proximity
    intensity_temperature = 6  # Strong focus on high-intensity fires
    suppressant_temperature = 4  # Balance suppressant allocation
    burnout_penalty_temperature = 5  # Prevent penalties from fires burning out
    reward_temperature = 4  # Emphasize the importance of rewards

    scores = []
    for i in range(len(fire_pos)):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos

        # --- Calculate distance score ---
        distance = math.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        distance_score = math.exp(-distance / distance_temperature)

        # --- Calculate fire intensity score ---
        intensity_score = math.exp(fire_intensities[i] / intensity_temperature)

        # --- Calculate suppressant efficiency score ---
        required_suppressant = fire_intensities[i] / agent_fire_reduction_power
        suppressant_score = (
            math.exp(-required_suppressant / suppressant_temperature)
            if agent_suppressant_num >= required_suppressant else 0
        )

        # --- Calculate burnout penalty score ---
        burnout_penalty = math.exp(fire_levels[i] / burnout_penalty_temperature)

        # --- Calculate reward weight score ---
        reward_score = math.exp(fire_putout_weight[i] / reward_temperature)

        # --- Combine scores ---
        total_score = (
            reward_score *
            intensity_score *
            suppressant_score /
            (burnout_penalty + distance_score + 1e-6)  # Prevent overly favoring distant fires
        )
        scores.append(total_score)

    # Choose the task with the highest score
    return int(scores.index(max(scores)))