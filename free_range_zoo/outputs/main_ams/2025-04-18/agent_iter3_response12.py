import numpy as np

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

    max_score = float('-inf')
    best_fire = None

    # Individual temperature factors for different score components
    temperature_distance = 0.1
    temperature_intensity = 0.5
    temperature_level = 0.7

    for i, (fire_position, fire_level, fire_intensity, fire_weight) in enumerate(zip(fire_pos, fire_levels, fire_intensities, fire_putout_weight)):

        # Normalize and compute the score components with respect to their individual temperature factors
        distance_score = np.exp(-np.linalg.norm(np.subtract(agent_pos, fire_position)) / temperature_distance)
        intensity_score = np.exp(fire_intensity / temperature_intensity)
        level_score = np.exp(fire_level / temperature_level)

        # Calculate suppression efficiency score considering agent's fire reduction power and available suppressant resources
        suppression_score = agent_fire_reduction_power * agent_suppressant_num / (fire_intensity + 1)

        # Compute overall score for the fire task considering the task weight, distance, intensity, level scores and suppression score
        task_score = fire_weight * distance_score * intensity_score * level_score * suppression_score

        if task_score > max_score:
            max_score = task_score
            best_fire = i

    return best_fire