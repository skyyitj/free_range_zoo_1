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
    fire_levels: List[int],                      # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    """
    Choose the optimal fire-fighting task for a single agent.

    Input Parameters:
        Agent Properties:
            agent_pos: (y, x) coordinates of the agent
            agent_fire_reduction_power: Fire suppression capability
            agent_suppressant_num: Available suppressant resources

        Team Information:
            other_agents_pos: List of (y, x) positions for all other agents
                            Shape: (num_agents-1, 2)

        Fire Information:
            fire_pos: List of (y, x) coordinates for all fires
                     Shape: (num_tasks, 2)
            fire_levels: Current fire intensity at each location
                        Shape: (num_tasks,)
            fire_intensities: Base difficulty of extinguishing each fire
                            Shape: (num_tasks,)

        Task Weights:
            fire_putout_weight: Priority weights for task selection
                               Shape: (num_tasks,)

    Returns:
        int: The index of the selected fire task (0 to num_tasks-1)
    """
    # Normalize distance score temperature
    distance_temperature = 0.5
    # Normalize reduction score temperature
    suppression_temperature = 0.3
    # Normalize reward score temperature
    reward_temperature = 0.7

    num_tasks = len(fire_pos)
    scores = []

    for i in range(num_tasks):
        # Calculate distance from agent to the fire location
        fire_y, fire_x = fire_pos[i]
        dist = np.sqrt((agent_pos[0] - fire_y) ** 2 + (agent_pos[1] - fire_x) ** 2)
        distance_score = np.exp(-dist / distance_temperature)  # Prefer closer fires

        # Calculate potential suppression effect on fire intensity
        remaining_fire = fire_intensities[i] - (agent_suppressant_num * agent_fire_reduction_power)
        suppression_score = np.exp(-remaining_fire / suppression_temperature)  # Prefer fires that can be suppressed effectively

        # Factor in priority weight of the fire location
        reward_score = np.exp(fire_putout_weight[i] / reward_temperature)  # Prefer higher-weight fires

        # Combine scores with multiplication
        combined_score = distance_score * suppression_score * reward_score
        scores.append(combined_score)

    # Select the fire with the maximum score
    selected_task = int(np.argmax(scores))
    return selected_task