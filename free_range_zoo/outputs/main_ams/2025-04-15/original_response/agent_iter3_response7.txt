import numpy as np
from typing import List, Tuple
from scipy.spatial import distance

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
    fire_intensities: List[float],               # Current intensity vale of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    intensity_temperature = 0.1
    distance_temperature = 0.01
    level_temperature = 0.3

    for task in range(num_tasks):
        # get euclidean distance between fire and agent
        fire_distance = distance.euclidean(agent_pos, fire_pos[task])
        
        # add a new factor that whether agent can put out the fire directly to guide the task assignment
        fire_putout_prob = 1 if fire_intensities[task] <= can_put_out_fire else 0
        # Calculate agent's distance to closest agent
        other_agents_distance = [distance.euclidean(agent_pos, pos) for pos in other_agents_pos]
        min_distance_to_agents = min(other_agents_distance) if other_agents_distance else 0
        
        # We update the score assignement function by considering the distance to the nearest agent
        scores[task] = (
            np.exp(-fire_levels[task]*level_temperature) +
            np.exp(-fire_intensities[task]/can_put_out_fire * intensity_temperature) *
            np.exp(-fire_distance * distance_temperature*fire_putout_prob) +
            np.exp(-min_distance_to_agents * distance_temperature*(1-fire_putout_prob))
            ) * fire_putout_weight[task]  
    # return the index of the task with the highest score
    max_score_task = np.argmax(scores)
    return max_score_task
# The policy function has been updated. Let's rerun the simulation to evaluate its performance.