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
    # === Fire Task Selection Scoring ===
    max_score = float('-inf')
    best_fire = None

    # Compute average fire intensity
    average_fire_intensity = sum(fire_intensities) / len(fire_intensities)

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):
        # Calculate euclidean distance from agent to each fire
        dist = ((fire_position[0]-agent_pos[0])**2 + (fire_position[1]-agent_pos[1])**2)**0.5
    
        # Add a bonus score for fires with intensity above average, to prioritize them
        intensity_bonus = 1.5 if fire_intensity > average_fire_intensity else 1

        # Score = Weight × Suppression Potential - ((Distance × Fire Intensity) × Intensity_Bonus)
        # The intensity bonus will give higher score to fires with above average intensity, hence prioritizing them
        score = fire_weight * (agent_suppressant_num * agent_fire_reduction_power/fire_intensity) - (dist * fire_intensity) * intensity_bonus
        if score > max_score:
            max_score = score
            best_fire = i

    return best_fire