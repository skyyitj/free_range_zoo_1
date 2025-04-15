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

    # Calculate distances to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Initialize the best task id and its associated highest score
    best_task = -1
    highest_score = -np.inf

    # If the agent does not have enough suppressant, return the default task id (-1)
    if agent_suppressant_num <= 0:
        return best_task

    # Calculate distance of each other agents to the fires
    other_agents_distance_to_fires = [[np.sqrt((fire[0]-pos[0])**2 + (fire[1]-pos[1])**2) for fire in fire_pos] for pos in other_agents_pos]

    # For each fire
    for task_id in range(num_fires):

        # Check if any other agent is closer to the fire
        is_other_agent_closer = any([distances[task_id] > agent_distance[task_id] for agent_distance in other_agents_distance_to_fires])

        # Skip this fire if another agent is closer to it.
        if is_other_agent_closer:
            continue

        # Calculate the efficiency score of the current task
        task_score = (fire_intensities[task_id] 
                      * agent_fire_reduction_power 
                      * agent_suppressant_num 
                      / distances[task_id]
                      - fire_levels[task_id])

        # If the current score is higher than the highest score
        if task_score > highest_score:
            # Set the current score to the highest score
            highest_score = task_score
            # Update the best task
            best_task = task_id

    # Return the index of the best task
    return best_task