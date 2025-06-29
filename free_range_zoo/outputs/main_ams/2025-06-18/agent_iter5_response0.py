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

    # Initialize temperature parameters for score calculations
    distance_temp = 10.0  # Temperature for distance scaling
    intensity_temp = 0.5  # Temperature for fire intensity scaling
    weight_temp = 1.0     # Temperature for priority weight scaling

    # Helper function to calculate Manhattan distance between two points
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    best_task_idx = 0  # Default to the first task
    highest_score = -float('inf')

    # Iterate through all fire tasks
    for i in range(len(fire_pos)):
        # Fire position and properties
        fire_y, fire_x = fire_pos[i]
        fire_level = fire_levels[i]
        fire_intensity = fire_intensities[i]
        fire_weight = fire_putout_weight[i]

        # Calculate distance between agent and fire
        distance = manhattan_distance(agent_pos, (fire_y, fire_x))

        # Calculate suppressant usage and remaining fire intensity
        suppressant_used = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        remaining_fire = fire_intensity - suppressant_used * agent_fire_reduction_power

        # Penalty for high remaining fire (inducing fire spread risk)
        fire_spread_risk = max(remaining_fire, 0)

        # Normalize and transform individual components into a score
        distance_score = -np.exp(-distance / distance_temp)
        intensity_score = np.exp(-fire_intensity / intensity_temp)
        weight_score = np.exp(fire_weight / weight_temp)

        # Combine the scores (weight them to reflect priority order)
        score = weight_score + intensity_score - fire_spread_risk + distance_score

        # Update the best task based on the score
        if score > highest_score:
            best_task_idx = i
            highest_score = score

    return best_task_idx