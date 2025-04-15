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
    fire_intensities: List[float],
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    """
    import numpy as np

    if agent_suppressant_num <= 0:
        # If the agent has no suppressant, it cannot fight any fire
        return -1
    
    suppressable_fires = [i for i, fire_level in enumerate(fire_levels) if fire_level < agent_fire_reduction_power]
    if not suppressable_fires:
        # If there are no suppressable fires, the agent cannot fight any fire
        return -1
    
    distance_to_fires = np.linalg.norm(np.array(agent_pos) - np.array(fire_pos), axis=1)
    
    # The agents should prioritize fires that are close and have high intensity
    fire_scores = [fire_intensities[i] / distance_to_fires[i] for i in suppressable_fires]
    
    # Choose the fire with the highest score
    chosen_fire = np.argmax(fire_scores)
    
    return chosen_fire