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
    num_tasks = len(fire_pos)
    best_task_index = -1
    highest_score = float('-inf')

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Optimization through calculation of actual suppressant usage vs available
        effective_suppressant_usage = min(fire_intensity / agent_fire_reduction_power, agent_suppressant_num)
        
        # Distance calculation and influence
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)
        
        # Importance weight directly incorporated as priority factor
        importance_weight = fire_putout_weight[task_index]
        
        # Score components considerations
        distance_influence = 5.0  # Smoother distance decrease (was too harsh before)
        suppressant_efficiency_temp = 0.5  # Encouraging better suppressant use with softer curve
        importance_influence = 2.0  # Doubling influence of task importance to prioritize critical tasks more

        # Recalibrating scoring function based on feedback analysis
        task_score = (
            -np.log1p(distance) / distance_influence +  # Making distance a less dramatic negative
            np.log1p(effective_supressant_usage) / suppressant_efficiency_temp +  # Smoother suppressant efficiency
            np.log1p(importance_weight) * importance_influence  # Increased weight for importance
        )
        
        # Determining the highest score as the best task index
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index