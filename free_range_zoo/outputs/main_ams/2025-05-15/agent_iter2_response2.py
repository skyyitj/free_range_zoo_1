import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.
    """
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Adjust distance_normalization_temp to a slightly lower value to ensure greater relevance of distance factor.
    # Intensity normalization temperature has been slightly increased to dampen the effect of high-intensity fires overpowering distance consideration.
    distance_normalization_temp = 0.05 
    intensity_normalization_temp = 0.1
    
    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        norm_distance = np.exp(-distance_normalization_temp * distance)
        
        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_normalization_temp * intensity)
        
        # Rebalancing of the score calculations to prioritize heavier fires while considering agent capability and remaining distances
        # Considering suppressant availability in the context of intensity levels to ensure sustainable resource management
        effective_suppression = min(agent_fire_reduction_power * agent_suppressant_num, intensity)
        suppressant_utilization_score = effective_suppression / agent_fire_reduction_power
        
        score = fire_putout_weight[i] * suppressant_utilization_score * norm_distance * norm_intensity
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index