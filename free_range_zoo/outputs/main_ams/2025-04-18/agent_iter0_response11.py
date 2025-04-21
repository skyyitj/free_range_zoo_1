import numpy as np

def single_agent_policy(
    agent_pos, 
    agent_fire_reduction_power, 
    agent_suppressant_num, 
    other_agents_pos, 
    fire_pos, 
    fire_levels, 
    fire_intensities, 
    fire_putout_weight):

    max_task_score = -np.inf  # initialize the max score to negative infinity
    selected_task_index = None  # initialize the selected task index to None

    num_tasks = len(fire_pos)  # total number of tasks

    # iterate over all tasks
    for task in range(num_tasks):
        # calculate the distance from the agent to the fire
        distance = np.sqrt((agent_pos[0] - fire_pos[task][0])**2 + (agent_pos[1] - fire_pos[task][1])**2)
        
        # avoid task if we cannot put out the fire due to insufficient resources or low power
        if agent_suppressant_num * agent_fire_reduction_power < fire_levels[task]:
            continue

        # calculate the normalized factors
        normalized_distance = (1/(distance+1e-10))  # adding small value to avoid division by zero
        normalized_fire_level = (1/(fire_levels[task]+1e-10))  # adding small value to avoid division by zero
        normalized_agent_power = (agent_fire_reduction_power/np.max(fire_levels))  # normalize by the max fire level
        normalized_suppressant_available = (agent_suppressant_num/np.max(fire_levels))  # normalize by the max fire level

        # calculate the task score considering all factors
        task_score = (fire_putout_weight[task]  
                      * normalized_distance
                      * normalized_fire_level
                      * normalized_agent_power
                      * normalized_suppressant_available)

        # if this task_score is better than the current maximum, update the max and the corresponding task index
        if task_score > max_task_score:
            max_task_score = task_score
            selected_task_index = task

    return selected_task_index