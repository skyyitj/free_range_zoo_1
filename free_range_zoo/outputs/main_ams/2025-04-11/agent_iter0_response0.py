import numpy as np

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
    
    # Calculate euclidean distances to all fires
    distances = [np.sqrt((agent_pos[0]-f[0])**2 + (agent_pos[1]-f[1])**2) for f in fire_pos]

    # Calculate number of suppressants required by each fire
    suppressants_required = [l / agent_fire_reduction_power for l in fire_levels]

    feasible_fires = [i for i in range(len(fire_pos)) if suppressants_required[i] <= agent_suppressant_num]

    # If no fires are feasible, return None
    if not feasible_fires:
        return None

    # Choose the most feasible fire: closest and with highest intensity
    chosen_fire = min(feasible_fires, key=lambda i: (distances[i], -fire_intensities[i]))
    
    return chosen_fire