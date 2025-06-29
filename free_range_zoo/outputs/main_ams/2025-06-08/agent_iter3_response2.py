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
    distance_temperature = 10       # Soft focus on proximity
    intensity_temperature = 6       # Emphasize high-intensity fires
    suppressant_temperature = 3     # Balance suppressant efficiency
    burnout_penalty_temperature = 5 # Penalize fires at risk of burning out
    reward_temperature = 4          # Prioritize high-reward fires
    suppression_potential_temperature = 2 # Target fires close to extinguishment

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

        # --- Calculate suppression potential score ---
        suppression_potential = (
            math.exp((agent_fire_reduction_power * agent_suppressant_num - fire_intensities[i]) / suppression_potential_temperature)
            if fire_intensities[i] <= (agent_fire_reduction_power * agent_suppressant_num) else 0
        )

        # --- Combine scores ---
        total_score = (
            reward_score *
            suppressant_score *
            intensity_score *
            suppression_potential *
            distance_score /
            (burnout_penalty + 1e-6)  # Prevent division by zero
        )
        scores.append(total_score)

    # Choose the task with the highest score
    return int(scores.index(max(scores)))