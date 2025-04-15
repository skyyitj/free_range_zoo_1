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

    # For each fire
    for task_id in range(num_fires):

        # Calculate the urgency score of the current task
        # Increase the weight of fire intensity and distance in the score
        task_score = (10*fire_intensities[task_id] 
                      * agent_fire_reduction_power 
                      * agent_suppressant_num 
                      / (1+distances[task_id])
                    - 0.5*fire_levels[task_id])

        # If the current score is higher than the highest score
        if task_score > highest_score:
            # Set the current score to the highest score
            highest_score = task_score
            # Update the best task
            best_task = task_id

    # Return the index of the best task
    return best_task