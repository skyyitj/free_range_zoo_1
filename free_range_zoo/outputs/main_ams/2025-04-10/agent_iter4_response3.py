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
    If no fires are present, return -1 (no task assigned).
    """

    # If there are no fires, we simply return -1 or idle task
    if not fire_levels:
        return -1  # No fires to handle, so the agent should do nothing

    # Find the fire with the highest intensity that the agent can handle
    best_fire_idx = -1
    max_fire_intensity = -1
    for i, fire in enumerate(fire_pos):
        if fire_levels[i] > 0 and fire_intensities[i] > max_fire_intensity:  # Only consider active fires
            # Prioritize fire with higher intensity (can also consider distance, if needed)
            best_fire_idx = i
            max_fire_intensity = fire_intensities[i]

    # Check if suppressant is available to tackle the fire
    if agent_suppressant_num > 0 and best_fire_idx != -1:
        # Apply fire suppression to the selected fire, using agent's fire suppression power
        fire_levels[best_fire_idx] -= agent_fire_reduction_power

    # If there are no fires left, return -1 (idle)
    return best_fire_idx