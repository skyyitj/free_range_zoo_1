import numpy as np
from typing import Tuple, List

# Calculate the Euclidean distance between two points
def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]): 
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Define the policy function
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

    # Number of tasks
    num_tasks = len(fire_pos)

    # Store the scores for every fire
    fire_scores = []

    # Temperature variables for score components transformations (self-defined constants)
    distance_temp = 0.1
    intensity_temp = 0.2
    level_temp = 0.3

    # Calculate the score for each fire
    for task in range(num_tasks):

        # Calculate the distance from the agent to the fire
        distance = euclidean_distance(agent_pos, fire_pos[task])

        # Transform distance to a more useful range
        distance_transformed = np.exp(-distance / distance_temp)

        # Transform the intensity to a more useful range
        intensity_transformed = np.exp(fire_intensities[task] / intensity_temp)

        # Transform the level to a more useful range
        level_transformed = np.exp(fire_levels[task] / level_temp)

        # Add the transformed scores to the total score for this task
        score = (fire_putout_weight[task] * level_transformed * intensity_transformed) / distance_transformed

        # Append the score to the list of all scores
        fire_scores.append(score)

    # Find the index of the highest scoring fire - this is the best fire for this agent to fight
    best_fire = np.argmax(fire_scores)

    # Return the index of the best fire
    return best_fire