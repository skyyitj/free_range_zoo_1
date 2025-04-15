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
    Determines the best action for an agent in the wildfire environment.

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
    
    # Filter for fires that are still active (level > 0)
    active_fires = [i for i, level in enumerate(fire_levels) if level > 0]
    
    if not active_fires:
        return -1  # No active fires to address
    
    # Prioritize the fires by intensity (higher intensity should be prioritized)
    best_fire_idx = None
    highest_intensity = -1
    
    for i in active_fires:
        # Check if this fire has a higher intensity
        if fire_intensities[i] > highest_intensity:
            best_fire_idx = i
            highest_intensity = fire_intensities[i]

    # Check if the agent has enough suppressant to tackle the selected fire
    if agent_suppressant_num >= fire_intensities[best_fire_idx]:
        # If enough suppressant, proceed with the most intense fire
        return best_fire_idx
    else:
        # If not enough suppressant, prioritize fires where the suppressant can have the highest efficiency
        best_fire_idx = max(active_fires, key=lambda i: fire_intensities[i] / agent_suppressant_num)
        return best_fire_idx