import numpy as np


def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:

    # Number of tasks
    num_tasks = len(fire_pos)
    
    # Initialize task_scores to store the performance of each task
    task_scores = np.zeros(num_tasks)
    
    # Temperature parameters for score normalization
    agent_suppressant_temp = 0.1
    fire_level_temp = 0.1
    fire_intensity_temp = 0.1
    fire_putout_weight_temp = 0.1
    
    # Determine the score of each task
    for task in range(num_tasks):

        # Calculate the Euclidean distance from the agent to the fire
        fire_distance = np.sqrt((agent_pos[0]-fire_pos[task][0])**2 + (agent_pos[1]-fire_pos[task][1])**2)

        # Calculate agent's influence on fire: suppressant availability × reduction power / distance
        agent_influence = (agent_suppressant_num * agent_fire_reduction_power) / fire_distance
        agent_influence = np.exp(agent_influence / agent_suppressant_temp)
        
        # Normalize the fire level
        normalized_fire_level = np.exp(-fire_levels[task] / fire_level_temp)
        
        # Normalize the fire intensity
        normalized_fire_intensity = np.exp(-fire_intensities[task] / fire_intensity_temp)
        
        # Normalize the fire putout weight
        normalized_fire_putout_weight = np.exp(fire_putout_weight[task] / fire_putout_weight_temp)
        
        # Score for the task is a function of agent influence + fire level + fire intensity + fire putout weight
        task_scores[task] = agent_influence + normalized_fire_level + normalized_fire_intensity + normalized_fire_putout_weight
    
    # Return the task with the highest score
    return np.argmax(task_scores)