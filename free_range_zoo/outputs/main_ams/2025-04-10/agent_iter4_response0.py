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
    
    # Check if there are fires to extinguish
    if not fire_levels or len(fire_levels) == 0:
        return -1  # No fires, so the agent does nothing (or returns some default action)

    # Filter out inactive fires (fires that have already gone out)
    active_fires = [(i, level) for i, level in enumerate(fire_levels) if level > 0]

    if not active_fires:
        return -1  # No active fires left to extinguish
    
    # Strategy: Prioritize fires with the highest intensity and ensure efficient use of suppressant
    best_fire_idx = None
    max_priority_score = -float('inf')  # Score to decide which fire to tackle
    
    for i, fire in active_fires:
        # Calculate a "priority score" based on fire intensity and fire level
        # Consider that the agent should prioritize both high intensity and high level fires
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        
        # Priority heuristic: prioritize based on a weighted combination of fire level and intensity
        priority_score = fire_intensity * fire_level / (fire_intensity + 1)  # Example of scoring strategy
        
        if priority_score > max_priority_score:
            max_priority_score = priority_score
            best_fire_idx = i
    
    # Return the index of the fire with the highest priority score
    return best_fire_idx