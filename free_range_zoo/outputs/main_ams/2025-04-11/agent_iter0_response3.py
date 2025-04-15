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

    # For each fire
    for task_id in range(num_fires):
        # Skip the task if the agent does not have enough suppressant
        if agent_suppressant_num <= 0:
            break

        # Calculate the efficiency score of the current task
        # The score is determined by 
        # (1) the intensity of the fire, 
        # (2) the agent's fire reduction power, 
        # (3) the agent's current suppressant, 
        # (4) the distance to the fire 
        # (5) and the potential punishment for not putting out the fire (which is equal to the fire level)
        # We want high intensity fire, close distance, high agent power and high suppressant count to all contribute positively to the score. 
        # Fire level contributes negatively as we want to avoid high level fire that is about to go out on its own.
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

        # Use one unit of suppressant on this task
        agent_suppressant_num -= 1

    # Return the index of the best task
    return best_task