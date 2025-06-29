def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    import numpy as np

    num_tasks = len(fire_pos)
    scores = []
    
    # Temperature parameters for normalization
    intensity_temperature = 1.0
    weight_temperature = 1.0
    distance_temperature = 1.0

    for i in range(num_tasks):
        # Distance factor: prioritize closer fires
        fire_y, fire_x = fire_pos[i]
        distance = np.sqrt((agent_pos[0] - fire_y)**2 + (agent_pos[1] - fire_x)**2)
        distance_score = np.exp(-distance / distance_temperature)
        
        # Intensity factor: prioritize higher intensity fires
        intensity_score = np.exp(fire_intensities[i] / intensity_temperature)
        
        # Reward weight: prioritize fires based on their importance (from fire_putout_weight)
        weight_score = np.exp(fire_putout_weight[i] / weight_temperature)
        
        # Calculate remaining suppressant utility for fire suppression
        estimated_fire_remaining = fire_intensities[i] - (agent_suppressant_num * agent_fire_reduction_power)
        suppressant_score = max(1.0 - max(0, estimated_fire_remaining), 0.0)  # Avoid negative scores
        
        # Total score based on weighted factors
        total_score = (distance_score * 0.4) + (intensity_score * 0.4) + (weight_score * 0.2) + suppressant_score
        scores.append(total_score)

    # Choose the fire with the highest score
    selected_task_idx = int(np.argmax(scores))
    return selected_task_idx