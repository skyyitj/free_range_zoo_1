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

    num_tasks = len(fire_levels)
    scores = np.zeros(num_tasks)

    can_put_out_fire = agent_suppressant_num * agent_fire_reduction_power

    # Adjusted temperature parameters
    level_temperature = 0.2     # Original: 0.1
    intensity_temperature = 0.2 # Original: 0.1
    distance_temperature = 0.02 # Original: 0.01

    for task in range(num_tasks):
        # Calculate fire distance to all agents
        all_distance = np.array([distance.euclidean(agent_pos, fp) for fp in fire_pos])
        # Minimum travel time to the fire for any agent
        min_agent_distance = all_distance.min()

        # Reward function is based on three components: 
        # (1) a function of fire level, 
        # (2) a function of fire intensity/firefighting capacity, 
        # (3) a function of distance, 
        # with higher weight for closer fires and fires of higher intensity and level. 
        score = (
            np.exp(-fire_levels[task]*level_temperature) +
            np.exp(-fire_intensities[task]/can_put_out_fire*intensity_temperature) -
            np.exp(min_agent_distance*distance_temperature)
        )
        # Update the score of this task by multiplying its weight
        scores[task] = fire_putout_weight[task] * score

    # return the index of the task with the highest score
    max_score_task = np.argmax(scores)
    return max_score_task