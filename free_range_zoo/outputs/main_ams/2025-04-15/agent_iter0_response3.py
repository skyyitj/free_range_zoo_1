def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],                 # Current position of the agent (y, x)
    agent_fire_reduction_power: float,              # How much fire the agent can reduce
    agent_suppressant_num: float,                   # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]],    # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],            # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                         # Current intensity level of each fire
    fire_intensities: List[float],                  # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],                # Priority weights for fire suppression tasks
) -> int:
    import numpy as np

    agent_pos = np.array(agent_pos)                  # Make it easier to vectorize arithmetic operations
    fire_pos = np.array(fire_pos)                    # Convert to vector for better computation
    
    dists_to_fires = np.linalg.norm(agent_pos - fire_pos, axis=1)       # Calculate distance of agent from each fire
    suppressant_needed = fire_intensities / agent_fire_reduction_power  # Calculate how much suppressant is needed to put each fire out
    
    # Specify temperature parameters for exponential scoring
    temp_dist, temp_intens, temp_weight = 0.1, 0.1, 0.1
    
    # Score tasks based on distance, intensity, and priority weight
    scores = (np.exp(-temp_dist * dists_to_fires) +
              np.exp(-temp_intens * suppressant_needed) +
              np.exp(temp_weight * np.array(fire_putout_weight))
             )
    
    # Filter out tasks that can't be completed with available resources
    feasible_tasks = np.where((suppressant_needed <= agent_suppressant_num) & (fire_levels > 0))[0]
    feasible_scores = scores[feasible_tasks]
    
    if len(feasible_scores)==0:   # All tasks are too hard or fire has already been put out
        return None

    else:  # Return highest-scoring feasible task
        idx_best_task = np.argmax(feasible_scores)
        return feasible_tasks[idx_best_task]