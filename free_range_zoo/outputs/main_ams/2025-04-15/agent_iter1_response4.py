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
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_pos)                    # Number of fires 
    scores = []                                  # Score for each fire
    
    # Temperature parameters for the exponential weighting
    dist_temperature = 1.5 # increasing this value as distant fires should not be ignored altogether
    level_temperature = 4.0 # increasing this value slightly to encourage handling high intensity fires 
    intensity_temperature = 0.8 # increasing this value to deal with fires with higher intensity
    weight_temperature = 1.8 # increasing to give more priority to fires with higher weights
    
    # Iterate over each task
    for i in range(num_tasks):
        # Calculate distance to fire
        dist = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        # Calculate potential effects of agent's suppressant on the fire intensity
        effect = agent_suppressant_num * agent_fire_reduction_power / max(fire_intensities[i], 1)
        # Calculate score based on distance, fire level and intensity, and task weight
        # Apply exponential transformation to each component to normalize and weight them
        # Scores are negative as we seek to minimize (closest fire, lowest level/intensity, highest weight)
        score = -np.exp(-dist/dist_temperature) \
                -np.exp(-fire_levels[i]/level_temperature) \
                -np.exp(-fire_intensities[i]/intensity_temperature) \
                +np.exp(fire_putout_weight[i]/weight_temperature) \
                -np.exp(-effect/intensity_temperature)
        scores.append(score)

    # Return the index of the task with maximum score
    # As scores are negative, argmin() gives the maximum score
    return np.argmin(scores)