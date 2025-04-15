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
    
    # Handle case where no fires are present
    if not fire_levels:
        return -1  # Return -1 to indicate no task available

    # Filter out tasks that are no longer active (fire_levels <= 0)
    active_fires = [
        i for i, level in enumerate(fire_levels) if level > 0
    ]
    
    # If there are no active fires, return -1 (no task)
    if not active_fires:
        return -1

    # Now, choose the fire with the highest intensity, prioritizing severity
    # (This assumes that higher intensity means more difficult to extinguish, and thus a higher priority)
    best_fire_idx = active_fires[0]
    max_intensity = fire_intensities[best_fire_idx]
    
    for i in active_fires:
        if fire_intensities[i] > max_intensity:
            best_fire_idx = i
            max_intensity = fire_intensities[i]
    
    # If there are multiple agents, coordination can be added here,
    # for now, the agent will focus on the highest intensity fire available.
    
    # If the agent doesn't have enough suppressant or the fire is very large, it might decide not to act
    if agent_suppressant_num < fire_intensities[best_fire_idx]:
        # If not enough suppressant, might decide to wait or choose a different strategy
        return -1  # Indicating the agent cannot handle this fire at the moment

    return best_fire_idx