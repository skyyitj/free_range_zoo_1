from typing import List, Tuple

def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float]
) -> int:
    """
    Determines the best action for an agent in the wildfire environment, optimizing suppressant efficiency and fire intensity reduction.

    Args:
        agent_pos: Position of this agent (y, x)
        agent_fire_reduction_power: Fire suppression power of this agent
        agent_suppressant_num: Number of suppressant available

        other_agents_pos: Positions of all other agents [(y1, x1), (y2, x2), ...]
        fire_pos: Positions of all fire tasks [(y1, x1), (y2, x2), ...]
        fire_levels: Current fire level of each task
        fire_intensities: Intensity (difficulty) of each task

    Returns:
        int: Index of the chosen task to address (0 to num_tasks-1)
    """
    
    # Check if there are any fires to address
    if not fire_levels:
        return -1  # No fires to address, return an invalid index
    
    # Filter for fires that are still active
    active_fires = [i for i, level in enumerate(fire_levels) if level > 0]
    
    if not active_fires:
        return -1  # No active fires to address
    
    # Calculate the efficiency and importance of each fire to maximize reward
    fire_scores = []
    for i in active_fires:
        # Fire score is a weighted combination of intensity, level, and suppressant required
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        
        # Scale the fire intensity with respect to the agent's suppressant
        if agent_suppressant_num > 0:
            suppressant_efficiency = fire_intensity / agent_suppressant_num
        else:
            suppressant_efficiency = 0

        # Combine fire intensity reduction and suppressant efficiency
        # The score is higher for more intense fires that can be suppressed efficiently
        fire_score = (fire_intensity * fire_level) / max(suppressant_efficiency, 1)  # Avoid division by zero
        fire_scores.append((i, fire_score))

    # Select the fire with the highest score
    best_fire_idx = max(fire_scores, key=lambda x: x[1])[0]
    
    # Ensure that the agent has enough suppressant to suppress the chosen fire
    if agent_suppressant_num >= fire_intensities[best_fire_idx]:
        return best_fire_idx
    else:
        # If not enough suppressant, choose the most intense fire within the agent's capability
        possible_fires = [i for i in active_fires if fire_intensities[i] <= agent_suppressant_num]
        if possible_fires:
            return max(possible_fires, key=lambda i: fire_intensities[i])
        else:
            return -1  # No fire can be suppressed with current suppressant