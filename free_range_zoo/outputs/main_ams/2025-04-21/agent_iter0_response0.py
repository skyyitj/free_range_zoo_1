# Import the necessary libraries
import numpy as np
from typing import List, Tuple

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    # Calculate the euclidean distance from the agent to each fire
    distances_to_fires = [np.sqrt((agent_pos[0] - fire[0])**2 + (agent_pos[1] - fire[1])**2) for fire in fire_pos]
    
    # Calculate the expected agent impact on each fire
    expected_fire_reduction = [min(agent_suppressant_num * agent_fire_reduction_power, intensity) 
                               for intensity in fire_intensities]
    
    # Initialize score for each fire
    scores = []

    # Iterate over the tasks (fires)
    for i in range(len(fire_pos)):
        # Calculate the score for this task (lower is better)
        score = distances_to_fires[i] - expected_fire_reduction[i] / fire_putout_weight[i]
        
        # Store the score
        scores.append(score)
    
    # Choose the fire with the lowest score
    chosen_fire = np.argmin(scores) 

    # Return the chosen fire
    return chosen_fire