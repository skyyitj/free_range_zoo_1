def single_agent_policy(
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
    import math
    
    # Constants for scoring
    distance_temp = 0.1
    intensity_temp = 1.5
    resource_temp = 2.0

    num_tasks = len(fire_pos)
    best_task = None
    max_score = float('-inf')

    # Iterate over each fire to assess which fire to tackle
    for i in range(num_tasks):
        # Calculate distance from agent to fire
        dx = agent_pos[1] - fire_pos[i][1]
        dy = agent_pos[0] - fire_pos[i][0]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Calculate fire intensity after potential agent intervention
        possible_remaining_intensity = fire_intensities[i] - agent_fire_reduction_power * min(agent_suppressant_num, fire_levels[i])
        possible_remaining_intensity = max(0, possible_remaining_intensity)
        
        # Calculating the cost-effectiveness of dealing with this fire
        distance_score = math.exp(-distance * distance_temp)
        intensity_score = math.exp(-possible_remaining_intensity * intensity_temp)
        resource_efficiency_score = math.exp(-agent_suppressant_num * resource_temp)

        # Calculate overall score (weighted sum)
        score = (fire_putout_weight[i] * intensity_score * distance_score * resource_efficiency_score)

        # Select the fire with the best score
        if score > max_score:
            max_score = score
            best_task = i
    
    return best_task