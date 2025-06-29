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
    import numpy as np
    
    # === Initialization ===
    num_tasks = len(fire_pos)  # Number of fire locations
    scores = []

    # === Parameters for Normalization ===
    intensity_temperature = 5.0  # Temperature for intensity normalization
    distance_temperature = 2.0   # Temperature for distance normalization
    weight_temperature = 1.0     # Temperature for priority weight integration

    # === Calculate Scores for Each Task ===
    for i in range(num_tasks):
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        fire_priority = fire_putout_weight[i]
        fire_position = fire_pos[i]
        
        # Distance Calculation
        distance = np.linalg.norm(np.array(agent_pos) - np.array(fire_position))
        
        # Agent Effectiveness
        suppressant_use = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        expected_fire_reduction = suppressant_use * agent_fire_reduction_power

        # Fire Spread Risk
        remaining_fire_intensity = fire_intensity - expected_fire_reduction
        spread_risk = remaining_fire_intensity if remaining_fire_intensity > 0 else 0

        # Scoring Components
        intensity_score = np.exp(-fire_intensity / intensity_temperature)
        distance_score = np.exp(-distance / distance_temperature)
        priority_score = np.exp(fire_priority / weight_temperature)
        spread_risk_score = np.exp(-spread_risk / intensity_temperature)

        # Overall Score
        score = (priority_score + spread_risk_score) * intensity_score * distance_score
        scores.append(score)

    # === Choose Task with Highest Score ===
    best_task = int(np.argmax(scores))
    return best_task