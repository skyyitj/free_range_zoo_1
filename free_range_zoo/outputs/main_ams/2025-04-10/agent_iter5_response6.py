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
    Determines the best action for an agent in the wildfire environment, optimizing for fire suppression
    and suppressant efficiency.

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
    
    # Filter for fires that are still active (i.e., fire levels are greater than 0)
    active_fires = [i for i, level in enumerate(fire_levels) if level > 0]
    
    if not active_fires:
        return -1  # No active fires to address
    
    # Define a threshold for efficient suppressant usage (e.g., suppressant efficiency >= 1)
    efficiency_threshold = 1.0  # This threshold can be adjusted based on further testing
    
    # List of potential fires sorted by their efficiency (fire intensity per suppressant)
    fire_efficiencies = [
        fire_intensities[i] / agent_suppressant_num if fire_intensities[i] > 0 else 0
        for i in active_fires
    ]
    
    # Find the fire with the highest efficiency
    best_fire_idx = max(active_fires, key=lambda i: fire_efficiencies[i])
    
    # If the selected fire has an acceptable efficiency, prioritize it
    if fire_efficiencies[best_fire_idx] >= efficiency_threshold:
        return best_fire_idx
    
    # Otherwise, we try to select the fire with the highest fire intensity reduction per suppressant used
    # Select the fire where the suppression power per suppressant is maximized
    best_fire_idx = max(active_fires, key=lambda i: fire_intensities[i] / agent_suppressant_num)
    
    # Return the index of the chosen fire task
    return best_fire_idx