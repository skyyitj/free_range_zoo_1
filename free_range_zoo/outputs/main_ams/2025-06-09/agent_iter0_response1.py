from typing import Tuple, List

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
    import numpy as np
    
    num_tasks = len(fire_pos)
    selected_task = 0
    max_score = float('-inf')
    
    # Temperature parameters for transformations
    distance_temperature = 0.1
    intensity_temperature = 0.5
    weight_temperature = 0.2
    resource_temperature = 0.3

    for task_index in range(num_tasks):
        # Calculate the distance to the fire
        fire_y, fire_x = fire_pos[task_index]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        
        # Fire related details
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        fire_weight = fire_putout_weight[task_index]

        # Estimating the resource needed to extinguish this fire this turn
        needed_resource = np.minimum(fire_intensity / agent_fire_reduction_power, agent_suppressant_num)

        # Score calculation
        distance_score = np.exp(-distance_temperature * distance)
        intensity_score = np.exp(-intensity_temperature * fire_intensity)
        weight_score = np.exp(weight_temperature * fire_weight)
        resource_score = np.exp(resource_temperature * (agent_suppressant_num - needed_resource))_score

        overall_score = distance_score * intensity_score * weight_score * resource_score
        
        # Update task selection based on score
        if overall_score > max_score:
            max_score = overall_score
            selected_task = task_index

    return selected_task