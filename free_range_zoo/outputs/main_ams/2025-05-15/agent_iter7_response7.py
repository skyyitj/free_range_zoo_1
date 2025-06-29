import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    best_task_score = float('-inf')
    selected_task_index = -1

    # Parameters to be fine-tuned according to the simulation results
    distance_importance = 1.0  # Increased importance of distance
    intensity_importance = 1.0  # Elevated importance of intensity
    suppressant_importance = 2.0  # Increased focus on suppressant conservation
    reward_importance = 1.0  # Balanced importance for reward
    
    # Normalize suppressant based on maximum expected value from parameters
    max_possible_suppressant = 10  # Example max value based on domain knowledge
    normalized_suppressant = agent_suppressant_num / max_possible_suppressant
    
    for i in range(num_tasks):
        ypos, xpos = fire_pos[i]
        
        # Calculate distance from agent to fire
        distance = np.sqrt((agent_pos[0] - ypos)**2 + (agent_pos[1] - xpos)**2)
        
        # Calculate suppression score considering distance, intensity and available suppressant
        score = (
            reward_importance * fire_putout_weight[i] +
            (-1 * distance_importance * distance) +  # More negative score for farther distances
            (-1 * suppressant_importance / (1 + normalized_suppressant)) +  # More conservation as suppressant depletes
            intensity_importance * (fire_levels[i] * fire_intensities[i])  # Emphasize higher and more intense fires
        )
        
        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index