Python
import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:

    # If the agent has no suppressant, return -1 to indicate the need for recharge
    if agent_suppressant_num <= 0:
      return -1

    # Compute Euclidean distance from the agent to every fire
    distances = [np.sqrt((fire[0]-agent_pos[0])**2 + (fire[1]-agent_pos[1])**2) for fire in fire_pos]

    # Create a score for every fire based on its distance, intensity and level
    # We want to minimize distance, and maximize intensity and level
    # Since we are going to find the maximum score, we invert the distance part of the score
    scores = [(1/distance)*fire_levels[i]*fire_intensities[i] for i, distance in enumerate(distances)]

    # Choose the fire with the maximum score
    chosen_fire_index = np.argmax(scores)
  
    return chosen_fire_index