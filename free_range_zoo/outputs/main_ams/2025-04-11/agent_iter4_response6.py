import numpy as np
  
  def single_agent_policy(agent_pos, agent_fire_reduction_power, agent_suppressant_num, other_agents_pos, fire_pos, fire_levels, fire_intensities):
      distances = np.sqrt(np.sum((np.array(fire_pos) - np.array(agent_pos))**2, axis=1))
      
      scores = np.zeros(len(fire_pos))
      
      for i, (fire_level, fire_intensity, distance) in enumerate(zip(fire_levels, fire_intensities, distances)):
          if fire_level/distance <= agent_suppressant_num:
              scores[i] = (fire_level / distance) * fire_intensity
              
      return np.argmax(scores)