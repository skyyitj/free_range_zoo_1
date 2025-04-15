import random

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
    if agent_suppressant_num <= 0:  # If the agent has no suppressant left, return -1 to recharge.
        return -1

    valid_fires = [i for i, fl in enumerate(fire_levels) if fl < fire_intensities[i] and fl <= agent_suppressant_num ] 
    if not valid_fires:  # If all fires are self-extinguishing or can't be fully extinguished with current suppressant, return -1.
        return -1

    # Calculate scores for each fire. 
    fire_scores = []
    for i in valid_fires:
        # Calculate the Euclidean distance from the agent to the fire.
        distance = ((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)**0.5
        # Increase the effect of distance.
        score = fire_intensities[i] /(distance**2 + 1)
        fire_scores.append((score, i))

    # Choose the fire with the highest score.
    task_to_address = max(fire_scores)[1]

    # Introduce some randomness to encourage exploration.
    if random.random() < 0.05:
        task_to_address = random.choice(valid_fires)

    return task_to_address