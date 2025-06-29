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
    
    # Constants and weights adjustments
    distance_temp = 1.0  # Increase sensitivity to distance
    effect_temp = 1.5    # Reduce the explosive growth due to effectiveness
    weight_temp = 0.7    # Slightly more aggressive weight scaling

    # Score for each task based on criteria
    for task_index in range(num_tasks):
        # Fire Position and Intensity
        fire = fire_pos[task_index]
        
        # Calculate distance from agent to fire
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)
        
        # Effectiveness of the agent on this fire task
        potential_suppress_effect = agent_fire_reduction_power * min(agent_suppressant_num, fire_intensities[task_index] / agent_fire_reduction_power)
        
        # Calculate the weighted attractiveness of tackling this fire based on potential effect and priority weight
        importance_weight = fire_putout_weight[task_index]
        
        # Weight intensities and levels to reflect urgency
        urgency_weight = (fire_intensities[task_index] / np.max(fire_intensities)) + \
                         (fire_levels[task_index] / np.max(fire_levels))
        
        # Task score calculation with adjusted factors
        task_score = (
            -np.exp(distance / distance_temp) +  # We want smaller distances
            np.exp(potential_suppress_effect / effect_temp) +
            np.exp(importance_weight / weight_temp) +
            np.exp(urgency_weight / effect_temp)  # Added urgency weight factor
        )
        
        # Choosing the task with the highest score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index
    
    return best_task_index