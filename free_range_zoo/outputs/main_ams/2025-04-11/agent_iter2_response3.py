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
    num_agents = len(other_agents_pos)

    # Calculate the distances to all fires
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Initialize the best task id and its associated highest score
    best_task = -1
    highest_score = -np.inf

    # If the agent does not have enough suppressant, it returns the default task id (-1)
    if agent_suppressant_num <= 0:
        return best_task

    # For each fire
    for task_id in range(num_fires):
        
        # Count how many agents are closer to the fire than the current agent
        num_closer_agents = sum(np.sqrt((fire[task_id][0]-other[0])**2 + (fire[task_id][1]-other[1])**2) < distances[task_id] 
                                 for other in other_agents_pos)
        
        # If there are fewer agents closer to the fire than the current agent
        if num_closer_agents < num_agents / 2:  # Adjust this parameter as needed

            # Calculate the efficiency score of extinguishing the current fire
            task_score = (fire_intensities[task_id] * agent_fire_reduction_power * agent_suppressant_num / distances[task_id]
                          - fire_levels[task_id])

            # If the current score is higher than the highest score
            if task_score > highest_score:
                # Set the current score to the highest score
                highest_score = task_score
                # Update the best task
                best_task = task_id

    # Return the index of the best task
    return best_task