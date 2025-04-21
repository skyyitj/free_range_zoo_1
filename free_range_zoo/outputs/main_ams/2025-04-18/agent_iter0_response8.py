import math

def single_agent_policy(
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                     # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Parameters:
        - fire_pos: Locations of fires (y, x)
        - fire_levels: Current fire intensity at each location
        - fire_intensities: Intensity values of the fire
        - fire_putout_weight: Task weight for prioritization

    Returns:
        int: Index of the selected fire task
    """
    
    # Temperature variables for scoring transformation (fine-tuning prioritization)
    intensity_temp = 1.0
    weight_temp = 1.0
    
    # Calculate task scores for each fire
    task_scores = []
    for i in range(len(fire_pos)):
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        task_weight = fire_putout_weight[i]
        
        # Calculate the potential reduction of fire intensity based on available suppressant
        fire_reduction = min(fire_intensity, agent_suppressant_num * agent_fire_reduction_power)
        
        # After reduction, calculate the remaining intensity
        remaining_intensity = fire_intensity - fire_reduction
        
        # Priority scoring: Combine intensity, fire level, and weight for each fire task
        fire_score = (remaining_intensity ** intensity_temp) * (task_weight ** weight_temp)

        # Penalty for high fire levels that are about to self-extinguish (if intensity is still positive)
        if remaining_intensity <= 0 and fire_level > 0:
            fire_score -= 5  # Penalize if fire will self-extinguish due to level threshold
        
        task_scores.append(fire_score)
    
    # Normalize scores using exponential transformation for better differentiation
    max_score = max(task_scores)
    exp_scores = [math.exp(score - max_score) for score in task_scores]  # Avoid overflow, shift scores

    # Select the fire task with the highest transformed score
    selected_task = exp_scores.index(max(exp_scores))
    
    return selected_task