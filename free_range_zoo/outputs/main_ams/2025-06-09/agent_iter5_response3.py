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

    # Temperatures adjusted to amplify critical parameters
    distance_temp = 0.7
    suppressant_efficiency_temp = 0.7
    importance_temp = 1.9

    # This fire putting out efficiency considers current suppressant left
    suppressant_remaining_scale = agent_suppressant_num

    for task_index in range(num_tasks):
        fire = fire_pos[task_index]
        distance = np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2)
        
        # If suppressant is running low, focus on the highest priority fires
        target_suppressant_use = min(fire_intensities[task_index] / agent_fire_reduction_power, agent_suppressant_num)
        possible_reduction = agent_fire_reduction_power * target_suppressant_use
        
        # Incorporating weighting scale emphasizing more on critical fires
        importance_weight = np.exp(fire_putout_weight[task_index] * importance_temp)
        
        # Adjust score computation
        task_score = (
            -np.log(distance + 1) / distance_temp +
            np.log(possible_reduction + 1) + 
            importance_weight * suppressant_remaining_scale * np.log(1 + target_suppressant_use / suppressant_efficiency_temp)
        )
        
        # Select the task with the maximum score after considering new conditions
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index
            
    return best_task_index