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
  
  # Calculate the distance of the agent to each fire
  distances_to_fire = np.sqrt(np.sum((np.array(fire_pos) - np.array(agent_pos))**2, axis=1))

  # Calculate the suppressant needed for each fire
  suppressant_needed = np.array(fire_levels) / agent_fire_reduction_power

  # Find those fires which are extinguishable with the current suppressant level
  extinguishable_fires = np.argwhere(suppressant_needed <= agent_suppressant_num).flatten()

  if len(extinguishable_fires) == 0:
    return -1 # recharge

  # Among the extinguishable fires, now choose the one which requires least travel, 
  # prioritizing higher intensity fires
  chosen_fire = sorted(extinguishable_fires, key=lambda fire: (distances_to_fire[fire], -fire_intensities[fire]))[0]

  return chosen_fire