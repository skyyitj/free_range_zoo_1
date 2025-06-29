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
    import numpy as np

    num_tasks = len(fire_pos)
    scores = []

    # Temperature parameters for control over scoring components
    distance_temp = 10.0  # Affects distance-based scoring
    intensity_temp = 1.0  # Affects fire intensity-based scoring
    reward_temp = 1.0     # Affects reward-based scoring

    for i in range(num_tasks):
        # Distance from agent to fire task
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        distance = np.sqrt((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2)
        distance_score = np.exp(-distance / distance_temp)

        # Fire intensity and remaining difficulty after suppression
        fire_intensity = fire_intensities[i]
        predicted_remaining_fire = max(
            fire_intensity - (agent_suppressant_num * agent_fire_reduction_power), 0
        )
        intensity_score = np.exp(-predicted_remaining_fire / intensity_temp)

        # Reward prioritization
        reward_score = np.exp(fire_putout_weight[i] / reward_temp)

        # Aggregate score based on components
        total_score = distance_score * intensity_score * reward_score
        scores.append(total_score)

    # Choose the fire task with the highest score
    best_task_index = int(np.argmax(scores))
    return best_task_index