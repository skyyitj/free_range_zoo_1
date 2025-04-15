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
    # Initialize an empty list to store each task's score
    task_scores = []

    # Find the number of fire tasks
    num_tasks = len(fire_pos)

    # Iterate through each fire task
    for task_index in range(num_tasks):
        # Calculate distance from the agent to the fire
        distance = ((agent_pos[0] - fire_pos[task_index][0]) ** 2 + 
                    (agent_pos[1] - fire_pos[task_index][1]) ** 2) ** 0.5
        
        # Calculate how much suppression the agent can apply to the fire
        possible_suppression = min(agent_suppressant_num, fire_levels[task_index] * fire_intensities[task_index])
        
        # Calculate the effectiveness of suppression
        suppression_effectiveness = possible_suppression / (fire_levels[task_index] * fire_intensities[task_index])

        # Calculate the agent's contribution compared to the other agents
        other_agents_distance = [((pos[0] - fire_pos[task_index][0]) ** 2 + 
                                  (pos[1] - fire_pos[task_index][1]) ** 2) ** 0.5
                                 for pos in other_agents_pos]
        agent_contribution = 1 / (1 + sum([1 / (d + 1e-5) for d in other_agents_distance]))
        
        # Calculate score for this task
        # Positive contributions: fire_putout_weight * suppression_effectiveness * agent_contribution
        # Negative contributions: distance to fire
        # We set a temperature parameter for the exponential transformation of the score
        temperature = 2.0
        score = (fire_putout_weight[task_index] * suppression_effectiveness * agent_contribution - distance)
        score = np.exp(score / temperature)

        # Append the score to the task_scores list
        task_scores.append(score)

    # Select the task with the highest score
    best_task = np.argmax(task_scores)

    return best_task