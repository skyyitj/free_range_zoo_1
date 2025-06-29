import math
from typing import Tuple, List

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    # Initialize a list to store scores for each fire task
    num_tasks = len(fire_pos)
    scores = [0.0] * num_tasks

    # Determine the assignment score for each possible fire task
    for i in range(num_tasks):
        # Distance from agent to fire, using Euclidean distance:
        distance = math.sqrt(
            (agent_pos[0] - fire_pos[i][0]) ** 2 + \
            (agent_pos[1] - fire_pos[i][1]) ** 2
        )
        
        # Effecacy metric accounting for suppressant and agent's power:
        efficacy = agent_suppressant_num * agent_fire_reduction_power
        
        # Fire strength that can be controlled:
        controllable_strength = efficacy / (fire_intensities[i] + 1)  # added +1 to avoid divide by zero
        
        # Priority accounting for high-level fires:
        priority_weight = fire_putout_weight[i]

        # Score calulation:
        # Note: Larger Scores indicate better tasks
        score = ((priority_weight * controllable_strength) / (distance + 1))  # added +1 to avoid divide by zero
        scores[i] = score
    
    # Select the fire task with the highest score
    best_task_index = max(range(num_tasks), key=lambda i: scores[i])
    return best_task_index

# Example:
agent_pos = (50.0, 50.0)
agent_fire_reduction_power = 7.5
agent_supressant_num = 100.0
other_agents_pos = [(60.0, 50.0), (20.0, 30.0)]
fire_positions = [(45.0, 55.0), (60.0, 50.0)]
fire_levels = [3, 2]
fire_intensities = [50.0, 40.0]
fire_putout_weight = [2.0, 3.0]

# Expected outcome should choose a single best fire location based on input factors
print(single_agent_policy(agent_pos, agent_fire_reduction_power, agent_supressant_num, other_agents_pos, 
                          fire_positions, fire_levels, fire_intensities, fire_putout_weight))