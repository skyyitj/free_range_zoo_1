import math

def single_agent_policy(agent_pos, agent_fire_reduction_power, agent_suppressant_num,
                        other_agents_pos, fire_pos, fire_levels, fire_intensities,
                        fire_putout_weight):
    # Constants for decision algorithms
    distance_temp = 0.05  # Affects decision sensitivity to distance
    intensity_temp = 0.1  # Affects decision sensitivity to fire intensity
    weight_temp = 0.2     # Affects decision sensitivity to put-out weights

    num_tasks = len(fire_pos)
    best_task = -1
    highest_score = float('-inf')

    # Iterate over all fire tasks
    for task_index in range(num_tasks):
        # Evaluate distance from agent to the fire
        fire_y, fire_x = fire_pos[task_index]
        agent_y, agent_x = agent_pos
        distance = math.sqrt((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2)

        # Exponential weighting based on the distance
        distance_effect = math.exp(-distance_temp * distance)
        
        # Intensity influences the fire's impact and extinction difficulty
        intensity = fire_intensities[task_index]
        intensity_effect = math.exp(intensity_temp * intensity)
        
        # The importance weight given to putting out this fire
        weight_effect = math.exp(weight_temp * fire_putout_weight[task_index])
        
        # Total score calculation using all effects
        score = distance_effect * intensity_effect * weight_effect
        
        # Update to choose the task with the highest score
        if score > highest_score:
            highest_score = score
            best_task = task_index

    return best_task