import numpy as np

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
    # === Task Scoring Parameters ===
    distance_temperature = 0.1   # Controls sensitivity to distance
    intensity_temperature = 0.5  # Controls sensitivity to fire intensity

    # Number of fires
    num_fires = len(fire_pos)

    # Initialize score list for each fire
    task_scores = np.zeros(num_fires)
    
    # Iterate over each fire
    for fire in range(num_fires):
        # Fire Position
        fire_y, fire_x = fire_pos[fire]
        
        # Compute Euclidean distance to fire
        distance_to_fire = np.hypot(agent_pos[0] - fire_y, agent_pos[1] - fire_x) 

        # Compute potential fire remaining if extinguished by agent
        potential_fire_remaining = max(0, fire_levels[fire] - agent_suppressant_num * agent_fire_reduction_power)
        
        # Compute potential suppressant remaining if agent fights this fire
        potential_suppressant_remaining = max(0, agent_suppressant_num - fire_levels[fire] / agent_fire_reduction_power)
        
        # Score for Agent to fight this fire:
        # Consider distance to fire (lower is better)
        # Consider fire intensity (higher is more urgent)
        # Consider potential fire remaining (lower is better)
        # Consider potential suppressant remaining (higher is better)
        # Also consider fire's putout weight (higher is better)
        # Each component is adjusted with a temperature to control its influence
        task_scores[fire] = (np.exp(-distance_to_fire * distance_temperature) +
                             np.exp(fire_levels[fire] * intensity_temperature) +
                             np.exp(-potential_fire_remaining * intensity_temperature) +
                             np.exp(potential_suppressant_remaining * intensity_temperature) +
                             fire_putout_weight[fire])

    # Return task (fire) with highest score
    return np.argmax(task_scores)