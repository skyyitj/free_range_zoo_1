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
    
    # Define temperature parameters for score components
    distance_temp = 0.5
    intensity_temp = 0.7
    weight_temp = 1.0

    num_tasks = len(fire_pos)
    agent_y, agent_x = agent_pos

    best_task_index = 0
    max_score = float("-inf")

    # Iterate over all fire tasks
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]

        # 1. Distance-based factor: Closer fires are prioritized
        distance = ((fire_y - agent_y)**2 + (fire_x - agent_x)**2)**0.5
        distance_score = -distance / distance_temp  # Normalize distance (negative for closer fires)

        # 2. Fire intensity-based factor: Higher intensity fires are prioritized
        intensity_score = fire_intensities[i] / intensity_temp  # Normalize intensity

        # 3. Reward weight factor: Fires with high priority weights are prioritized
        weight_score = fire_putout_weight[i] / weight_temp  # Normalize weight

        # Combine score factors into a final score
        total_score = distance_score + intensity_score + weight_score

        # Check for best task based on computed score
        if total_score > max_score:
            max_score = total_score
            best_task_index = i

    return best_task_index