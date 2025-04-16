import numpy as np
from typing import Tuple, List

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Input Parameters:
        Agent Properties:
            agent_pos: (y, x) coordinates of the agent
            agent_fire_reduction_power: Fire suppression capability
            agent_suppressant_num: Available suppressant resources
  
        Team Information:
            other_agents_pos: List of (y, x) positions for all other agents
                            Shape: (num_agents-1, 2)
  
        Fire Information:
            fire_pos: List of (y, x) coordinates for all fires
                     Shape: (num_tasks, 2)
            fire_levels: Current fire intensity at each location
                        Shape: (num_tasks,)
            fire_intensities: Base difficulty of extinguishing each fire
                            Shape: (num_tasks,)
  
        Task Weights:
            fire_putout_weight: Priority weights for task selection
                               Shape: (num_tasks,)

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """

    scores = []
    num_tasks = len(fire_pos)

    # Temperature for distance decay
    dist_temp = 1.0
    for task in range(num_tasks):
        # Distance from agent to the fire task
        distance = np.sqrt((agent_pos[0]-fire_pos[task][0])**2 + (agent_pos[1]-fire_pos[task][1])**2)

        # Calculate the inverse of distance and fire intensity using softmax function and distance temperature
        distance_inv = 1/(1+np.exp(distance/dist_temp))
        intensity_inv = 1/(1+np.exp(fire_levels[task]))

        # Calculate the potential contribution of the agent to control the fire using its remaining suppressant
        contribution = min(agent_suppressant_num, fire_intensities[task]) * agent_fire_reduction_power

        # Final score of fire task including reward weights
        score = (distance_inv + intensity_inv + contribution) * fire_putout_weight[task]
        scores.append(score)
    
    return np.argmax(scores)     # return the index of the fire task with highest score