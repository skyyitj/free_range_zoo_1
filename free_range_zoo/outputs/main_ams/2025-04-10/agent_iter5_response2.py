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
    
    # Filter for fires that are still active (non-zero intensity)
    active_fires = [i for i, level in enumerate(fire_levels) if level > 0]
    
    if not active_fires:
        return -1  # No active fires to address
    
    # Calculate suppressant efficiency for each fire
    fire_efficiencies = []
    for i in active_fires:
        # Calculate how efficient suppressing this fire is, relative to intensity and available suppressant
        efficiency = fire_intensities[i] / agent_suppressant_num if agent_suppressant_num > 0 else 0
        fire_efficiencies.append((i, efficiency))
    
    # Sort fires based on highest suppressant efficiency, preferring higher efficiency
    best_fire_idx, _ = max(fire_efficiencies, key=lambda x: x[1])
    
    # Check if the agent has enough suppressant to deal with the selected fire
    if agent_suppressant_num >= fire_intensities[best_fire_idx]:
        return best_fire_idx
    else:
        # If not enough suppressant, prioritize smaller fires or those that are easier to suppress
        # with the available amount of suppressant
        best_fire_idx = min(active_fires, key=lambda i: fire_intensities[i])
        return best_fire_idx