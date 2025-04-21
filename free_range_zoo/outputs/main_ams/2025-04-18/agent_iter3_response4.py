from scipy.spatial import distance

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

    # === Scoring Fire Tasks ===
    min_score = float('inf')
    best_fire = None

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Distance-based factor for task selection
        dist = distance.euclidean(agent_pos, fire_position) / agent_suppressant_num

        # Risk factor calculation considering the fire level and intensity
        # Here, we modified the risk factor calculation to balance between suppressing high and low intensity fires
        risk_factor = (fire_level * fire_intensity) / (agent_fire_reduction_power * agent_suppressant_num)

        # Score calculation considering prioritization weight, risk factor, and distance factor
        score = (fire_weight * dist * risk_factor)

        if score < min_score:
            min_score = score
            best_fire = i

    return best_fire