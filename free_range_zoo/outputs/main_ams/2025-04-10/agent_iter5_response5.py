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
    Determines the best action for an agent in the wildfire environment, focusing on efficient fire suppression.

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
    
    # Prioritize fires with high intensity for more efficient suppression
    prioritized_fires = sorted(active_fires, key=lambda i: fire_intensities[i], reverse=True)
    
    # Attempt to minimize suppressant usage by choosing the most intense fire the agent can suppress
    for i in prioritized_fires:
        # Calculate suppressant usage for this fire
        required_suppressant = fire_intensities[i] * agent_fire_reduction_power
        
        if agent_suppressant_num >= required_suppressant:
            # If the agent has enough suppressant, choose this fire
            return i
        else:
            # If the agent doesn't have enough suppressant, attempt partial suppression
            # Choose fires based on maximum suppression per unit of suppressant
            return max(prioritized_fires, key=lambda i: fire_intensities[i] / agent_suppressant_num)
    
    # If no suitable fire can be suppressed, return an invalid index
    return -1