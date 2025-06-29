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
    """
    import math

    def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    num_tasks = len(fire_pos)
    scores = []

    # Temperature parameters for score component transformations.
    intensity_temp = 1.0  # Temperature for fire intensity transformation.
    distance_temp = 1.0   # Temperature for distance transformation.
    reward_temp = 1.0     # Temperature for reward weight transformation.

    # Iterate over all fire tasks.
    for i in range(num_tasks):
        # Calculate the estimated remaining fire intensity if the agent chooses this task.
        suppressant_used = min(agent_suppressant_num, fire_intensities[i] / agent_fire_reduction_power)
        remaining_fire_intensity = fire_intensities[i] - (suppressant_used * agent_fire_reduction_power)

        # Score component 1: Normalize fire intensities (high remaining intensity -> lower score).
        intensity_score = math.exp(-remaining_fire_intensity / intensity_temp)

        # Score component 2: Account for proximity (closer fires -> higher score).
        distance_to_fire = euclidean_distance(agent_pos, fire_pos[i])
        distance_score = math.exp(-distance_to_fire / distance_temp)

        # Score component 3: Use task prioritization weights (higher weight -> higher score).
        reward_score = math.exp(fire_putout_weight[i] / reward_temp)

        # Final weighted score combines all components.
        total_score = intensity_score * reward_score * distance_score
        scores.append(total_score)

    # Select the task with the highest score.
    return scores.index(max(scores))