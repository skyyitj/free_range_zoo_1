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
    import math

    num_tasks = len(fire_pos)
    task_scores = []

    # Temperature parameters for score components
    proximity_temp = 3.0     # Adjusted for stronger emphasis on distance
    severity_temp = 1.5      # Emphasizes fire intensity
    priority_weight_temp = 0.8

    for i in range(num_tasks):
        # Proximity component based on Euclidean distance
        agent_y, agent_x = agent_pos
        fire_y, fire_x = fire_pos[i]
        distance = math.sqrt((agent_y - fire_y) ** 2 + (agent_x - fire_x) ** 2)
        normalized_proximity = math.exp(-distance / proximity_temp)

        # Fire severity component based on fire intensity and level
        fire_severity = fire_intensities[i] * fire_levels[i]
        normalized_severity = math.exp(-fire_severity / severity_temp)

        # Task priority component
        normalized_priority_weight = math.exp(fire_putout_weight[i] / priority_weight_temp)

        # Extinguishable component (if the agent can fully extinguish the fire)
        extinguishable = agent_fire_reduction_power * agent_suppressant_num >= fire_intensities[i]
        extinguishable_score = 1.0 if extinguishable else 0.0

        # Combined score calculation
        task_score = (
            normalized_priority_weight
            + 3.0 * extinguishable_score
            - 2.0 * normalized_proximity
            + 1.5 * fire_levels[i] * normalized_severity
        )
        task_scores.append(task_score)

    # Select the best task (highest score) while ensuring suppressant availability
    if agent_suppressant_num > 0:
        best_task_idx = max(range(num_tasks), key=lambda idx: task_scores[idx])
    else:
        best_task_idx = -1  # No suppressant, no task assignment

    return best_task_idx