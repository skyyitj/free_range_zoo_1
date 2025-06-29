import numpy as np

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
    
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    
    # Coefficients for score calculation (scaling factors)
    distance_scale = 10.0
    level_scale = 1.0
    intensity_scale = 0.5
    weight_scale = 2.0
    suppressant_scale = 0.3
    
    for i in range(num_tasks):
        # Calculate distance from agent to fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance / distance_scale)

        # Normalize fire levels and intensities
        norm_level = fire_levels[i] / level_scale
        norm_intensity = np.exp(-fire_intensities[i] / intensity_scale)
        
        # Normalize fire weights
        norm_weight = fire_putout_weight[i] ** weight_scale
        
        # Effective suppressant capability
        norm_suppressant = np.exp(agent_suppressant_num / suppressant_scale)

        # Task score based on distance, fire intensity, and priority weight
        score = norm_distance * norm_level * norm_intensity * norm_weight * norm_supressant
        scores[i] = score

    # Select the highest scoring task
    best_task_idx = np.argmax(scores)
    return best_task_idx