def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                     # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    
    import numpy as np 
    # Parameters for score transformations
    distance_temperature = 1.0
    suppression_temperature = 1.0

    num_fires = len(fire_pos)
    scores = np.zeros(num_fires)

    # Compute the Euclidean distance from agent to each fire
    distances = [np.sqrt((agent_pos[0] - fpos[0])**2 + (agent_pos[1] - fpos[1])**2) for fpos in fire_pos]

    for i in range(num_fires):
        # Define the expected suppression based on agent's suppressant and reduction power
        expected_suppression = min(fire_levels[i], agent_fire_reduction_power * agent_suppressant_num)

        # Compute a distance score based on inverse distance (closer fires get higher scores)
        # And apply an exponential transformation for better differentiation
        distance_score = np.exp(-distances[i] / distance_temperature)

        # Compute a suppression score based on the expected fire suppression
        # Apply an exponential transformation for better differentiation
        suppression_score = np.exp(expected_suppression / suppression_temperature)

        # Calculate the total score as a weighted sum of the distance and suppression scores
        # Also include the fire's priority weight in the score
        scores[i] = fire_putout_weight[i] * (0.6 * distance_score + 0.4 * suppression_score)

    # Choose the task with the highest score
    chosen_task_idx = np.argmax(scores)

    return chosen_task_idx