import numpy as np

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    # === Task Prioritization ===
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    
    # Temperature parameters for weighted components
    distance_temperature = 0.1
    intensity_temperature = 0.2
    weight_temperature = 0.5
    level_temperature = 0.3
    
    agent_y, agent_x = agent_pos
    
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        
        # Calculate the distance to the fire
        distance_to_fire = np.sqrt((fire_y - agent_y)**2 + (fire_x - agent_x)**2)
        
        # Get other components
        fire_weight = fire_putout_weight[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        
        # Formulate the score based on:
        # - distance (lower is better)
        # - intensity (higher needs more attention)
        # - fire level (higher may mean more dangerous)
        # - the weight from externally defined importance (higher is higher priority)
        
        # Transform inputs to exponential scale to normalize
        distance_score = np.exp(-distance_to_fire / distance_temperature)
        intensity_score = np.exp(fire_intensity / intensity_temperature)
        weight_score = np.exp(fire_weight / weight_temperature)
        level_score = np.exp(fire_level / level_temperature)
        
        # Effective fire suppression indicator (assuming max use of agent's resources)
        suppression_potential = agent_fire_reduction_power * agent_suppressant_num
        suppression_score = suppression_potential / (fire_intensity + 1)
        
        # Compile scores
        scores[i] = (distance_score + intensity_score + weight_score + level_score + suppression_score)

    # Select the task with the highest score
    selected_task = np.argmax(scores)
    
    return selected_task