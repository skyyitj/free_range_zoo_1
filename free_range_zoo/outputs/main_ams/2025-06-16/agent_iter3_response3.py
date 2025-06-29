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
    import numpy as np

    num_tasks = len(fire_pos)
    best_task = -1
    best_score = -float('inf')

    # Temperature parameters for score transformations
    distance_temperature = 10.0
    intensity_temperature = 5.0
    reward_temperature = 2.0

    for i in range(num_tasks):
        # === Calculate Distance to Fire ===
        dist = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos[i]))
        distance_score = -np.exp(-dist / distance_temperature)  # Higher distance → lower score

        # === Calculate Fire Intensity Impact ===
        fire_remaining = fire_intensities[i] - (agent_fire_reduction_power * agent_suppressant_num)
        fire_intensity_score = -np.exp(fire_remaining / intensity_temperature)  # Higher remaining fire → lower score

        # === Adjust for Reward Weight ===
        reward_score = np.exp(fire_putout_weight[i] / reward_temperature)  # Higher reward → higher score

        # === Aggregate Task Score ===
        score = reward_score + intensity_score + distance_score

        # === Update the Best Task ===
        if score > best_score and fire_remaining > 0:  # Ensure task is valid
            best_score = score
            best_task = i

    return best_task