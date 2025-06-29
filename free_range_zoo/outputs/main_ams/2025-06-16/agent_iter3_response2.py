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
    
    # Temperature parameters for transforming distance, resource cost, and fire weight
    distance_temp = 1.0
    suppressant_temp = 1.0
    priority_temp = 1.0

    # Pre-compute distance from agent to each fire
    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    distances = np.array([euclidean_distance(agent_pos, fire) for fire in fire_pos])

    # Compute the expected reduction in fire intensity at each fire location
    expected_reductions = agent_fire_reduction_power * agent_suppressant_num

    # Calculate the remaining fire intensity after assigning the agent
    remaining_intensities = np.maximum(0, np.array(fire_intensities) - expected_reductions)
    
    # Calculate distance-based scores (inverse relationship: closer fires are prioritized)
    distance_scores = np.exp(-distances / distance_temp)

    # Compute suppressant cost normalization (inverse relationship: more suppressant cost reduces priority)
    suppressant_scores = np.exp(-remaining_intensities / suppressant_temp)

    # Normalize priority weights
    priority_scores = np.exp(np.array(fire_putout_weight) / priority_temp)

    # Combine scores for a final score for each fire location
    total_scores = distance_scores * suppressant_scores * priority_scores

    # Select the index of the fire task with the highest score
    selected_task = int(np.argmax(total_scores))

    return selected_task