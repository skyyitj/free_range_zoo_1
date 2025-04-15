import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float], 
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]], 
    fire_levels: List[float], 
    fire_intensities: List[float]
) -> int:

    num_fires = len(fire_pos)
    num_agents = len(other_agents_pos) + 1

    # Calculate distances to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Calculate the number of agents dealing with each fire
    agent_fire_counts = [0] * num_fires
    for other_agent_pos in other_agents_pos:
        nearest_fire_index = np.argmin([np.sqrt((fire[0]-other_agent_pos[0])**2 + (fire[1]-other_agent_pos[1])**2) for fire in fire_pos])
        agent_fire_counts[nearest_fire_index] += 1

    # Initialize the best task id and its associated highest score
    best_task = -1
    highest_score = -np.inf

    # If the agent does not have enough suppressant, return the default task id (-1)
    if agent_suppressant_num <= 0:
        return best_task

    # For each fire
    for task_id in range(num_fires):

        # Calculate the efficiency score of the current task
        task_score = (fire_intensities[task_id] 
                      * agent_fire_reduction_power 
                      * agent_suppressant_num 
                      / distances[task_id]
                      - fire_levels[task_id]
                      - agent_fire_counts[task_id] / num_agents) # We subtract the number of agents currently dealing with the fire divided by total agents

        # If the current score is higher than the highest score
        if task_score > highest_score:
            # Set the current score to the highest score
            highest_score = task_score
            # Update the best task
            best_task = task_id

    # Return the index of the best task
    return best_task