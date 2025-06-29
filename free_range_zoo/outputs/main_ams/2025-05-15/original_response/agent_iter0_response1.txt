import numpy as np

def single_agent_policy(
    agent_pos,
    agent_fire_reduction_power,
    agent_suppressant_num,
    other_agents_pos,
    fire_pos,
    fire_levels,
    fire_intensities,
    fire_putout_weight
):
    num_fires = len(fire_pos)
    
    scores = []
    
    distance_temperature = 0.1       # Controls sensitivity to distance
    intensity_temperature = 0.05     # Controls sensitivity to fire intensity
    level_temperature = 0.2          # Controls sensitivity to fire level
    weight_temperature = 1.0         # Controls sensitivity to task weights
    suppressant_temperature = 0.3    # Controls how the amount of suppressant influences choice

    # Calculate potential leftover suppressant factor
    suppressant_factors = [
        np.exp(-((f_intensity / agent_fire_reduction_power) / agent_suppressant_num) / suppressant_temperature)
        for f_intensity in fire_intensities
    ]

    for i in range(num_fires):
        fire_y, fire_x = fire_pos[i]
        agent_y, agent_x = agent_pos
        
        # Calculate distance between agent and fire
        distance = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        inv_distance = 1.0 / (1.0 + distance)
        
        # Calculate score components
        normalized_distance_score = np.exp(-distance * distance_temperature)
        intensity_score = np.exp(-fire_intensities[i] * intensity_temperature)
        level_score = np.exp(-fire_levels[i] * level_temperature)
        weight_score = np.exp(fire_putout_weight[i] * weight_temperature)
        suppressant_score = suppressant_factors[i]

        # Composite score for decision-making
        score = (
            normalized_distance_score * 
            intensity_score * 
            level_score * 
            weight_score *
            suppressant_score
        )

        scores.append(score)

    # Choose fire with the highest score
    best_fire_idx = np.argmax(scores)
    
    return best_fire_idx

# Example use case setup: Initialize variables
agent_position = (2.5, 2.5)
fire_reduction_power = 1.0
suppressant_amount = 5.0
other_agent_positions = [(1.0, 1.0), (3.0, 3.0)]
fire_positions = [(2.0, 2.0), (4.0, 4.0)]
fire_levels = [3, 5]
fire_intensities = [2.0, 3.5]
fire_weights = [1.0, 0.5]

# Run policy
fire_to_target = single_agent_policy(
    agent_position,
    fire_reduction_power,
    suppressant_amount,
    other_agent_positions,
    fire_positions,
    fire_levels,
    fire_intensities,
    fire_weights
)

print(f"Agent should target fire index: {fire_to_target}")