import numpy as np

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
    """
    Choose the optimal fire-fighting task for a single agent.
    """

    # Initialize number of tasks
    num_tasks = len(fire_pos)
    
    if num_tasks == 0:
        return 0  # If no tasks, no task can be selected

    # Priority scores for each fire based on various criteria
    scores = np.zeros(num_tasks)

    # Determine a scale for distance in weighing priorities
    distance_temperature = 100.0
    reduction_power_temperature = 2.0

    # Calculate the effectiveness of agent's contribution to each fire
    for i in range(num_tasks):
        # Simple Euclidean distance to each fire
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0]) ** 2 + (agent_pos[1] - fire_pos[i][1]) ** 2)
        # Normalized distance score (smaller is better)
        normalized_distance = np.exp(-distance / distance_temperature)

        # Estimated power application: Importance decreases with lower possibility to put out the fire completely
        effective_power = np.minimum(agent_suppressant_num * agent_fire_reduction_power, fire_intensities[i])
        effective_power_score = np.exp(effective_power / reduction_power_temperature)

        # The priority weight based on the importance of the task
        weight_score = fire_putout_weight[i]

        # Aggregate the influence of all factors: smaller distance, higher effectiveness, and higher priority
        scores[i] = (2 * normalized_distance + 2 * effective_power_score + 3 * weight_score)

    # Return the index of the task with the highest score
    return int(np.argmax(scores))