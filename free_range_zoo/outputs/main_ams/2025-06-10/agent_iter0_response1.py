import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],
    fire_putout_weight: List[float]
) -> int:
    
    # Constants/Separate temp for scaling factors which shouldn't be input parameters
    distance_temp = 0.1
    intensity_temp = 0.2
    weight_temp = 0.5
    capacity_temp = 0.3
    
    num_tasks = len(fire_pos)
    scores = np.zeros(num_tasks)
    
    for task_index in range(num_tasks):
        # Extract task information
        fire_location = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        task_weight = fire_putout_weight[task_index]
        
        # Compute the distance between the agent and the fire
        distance = np.sqrt((agent_pos[0] - fire_location[0]) ** 2 + (agent_pos[1] - fire_location[1]) ** 2)
        distance_factor = np.exp(-distance_temp * distance)
        
        # Consider fire intensity and how much an agent can potentially reduce it
        impact_potential = min(fire_intensity, agent_fire_reduction_power * agent_suppressant_num)
        impact_factor = np.exp(-intensity_temp * (fire_intensity - impact_potential))
        
        # Weight task prioritization by the suppression weight
        weight_factor = np.exp(weight_temp * task_weight)
        
        # Consider remaining agent suppressant capacity
        if agent_suppressant_num <= 0:
            capacity_factor = 0  # No capacity to fight fire
        else:
            capacity_factor = np.exp(capacity_temp * (agent_suppressant_num / fire_intensity))
        
        # Calculate overall score for this task
        scores[task_index] = distance_factor * impact_factor * weight_factor * capacity_factor

    # Select the fire task with the highest score
    selected_task_index = np.argmax(scores)
    
    return selected_task_index