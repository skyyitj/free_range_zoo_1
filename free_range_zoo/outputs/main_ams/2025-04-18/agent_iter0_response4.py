import math
from typing import List, Tuple

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    num_tasks = len(fire_pos)

    # Computes the Eucledian distance between two points.
    def compute_distance(pos1, pos2):
        y1, x1 = pos1
        y2, x2 = pos2
        return ((y1-y2)**2 + (x1-x2)**2)**0.5

    # Index of the best task.
    best_task_index = -1

    # Maximum score found so far.
    max_score = float('-inf')

    # For each fire, compute a score and keep track
    # of the fire with the highest score.
    for task_index in range(num_tasks):
        # The distance from the agent to the fire.
        dist = compute_distance(agent_pos, fire_pos[task_index])

        # Normalized intensity indicating how severe the fire is.
        intensity = fire_intensities[task_index] / max(fire_levels)

        # How much of the fire we can ideally suppress.
        potential_suppression = agent_suppressant_num * agent_fire_reduction_power

        # Calculate a score which factors in fire severity, the distance 
        # to the fire, how much we can suppress the fire and the fire's weight.
        # The score is inversely proportional to the distance and directly 
        # proportional to the other factors.
        score = fire_putout_weight[task_index] * potential_suppression * intensity / (dist + 1)

        # If this score is the highest, store it and the task index.
        if score > max_score:
            max_score = score
            best_task_index = task_index

    return best_task_index