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
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        
        # Improved calculation of distance with normalization
        distance = np.sqrt((agent_pos[0] - fire[0]) ** 2 + (agent_pos[1] - fire[1]) ** 2)
        
        # Consider both the intensity and the agent's suppression capabilities
        # Ensure not to waste resources for minor fires that other agents can handle easily
        expected_suppression = min(fire_intensity, agent_fire_reduction_power * agent_suppressant_num)
        potential_effectiveness = (expected_suppression / fire_intensity) * 100
        
        importance_weight = fire_putout_weight[task_index]
        
        # Adjustment of weight parameters
        distance_temp = 20.0  # Making distance less penalizing
        effectiveness_temp = 0.4  # Emphasis on efficiency of suppression
        importance_temp = 2.0  # Increased influence from task importance

        # Forming a balanced score
        task_score = (
            -np.log1p(distance / distance_temp) +
            np.log1p(potential_effectiveness / effectiveness_temp) +
            (importance_weight / importance_temp)
        )
        
        # Finding the task with the maximum score
        if task_score > highest_score:
            highest_score = task_score
            best_task_index = task_index

    return best_task_index