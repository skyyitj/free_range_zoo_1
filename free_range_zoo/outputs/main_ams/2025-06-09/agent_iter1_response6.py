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
    
    # Refine the scoring system by considering cumulative fire risk reduction and more weight sensitiveness
    for task_index in range(num_tests):
        # Fire Position and Intensity
        fire = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Calculate Euclidean distance
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)
        
        # Calculate the suppression power vs the intensity
        potential_suppress_effect = min(agent_fire_reduction_power * agent_suppressant_num, fire_intensity)
        
        # Obtain the weight intensity
        importance_weight = fire_putout_weight[task_index]
        
        # Adjust temperature parameters
        # Scaled down the influence of distance and increased the emphasis on weighting importance
        distance_temp = 3.0
        effect_temp = 1.0
        weight_temp = 0.2
        
        # Improvised score computation
        task_score = (
            -np.exp(distance / distance_temp) +  # Lower distances should still be favoured
            np.exp((potential_suppress_effect / effect_temp)) +
            np.exp((importance_weight / weight_temp) * (1 / max(0.01, distance)))  # Far distances reduces weight importance
        )
        
        # Select the task with the maximum score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    # Ensure a selected task if possible
    return best_task_index if best_task_index != -1 else 0