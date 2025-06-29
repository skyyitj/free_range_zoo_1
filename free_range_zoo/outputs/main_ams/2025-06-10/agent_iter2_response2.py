import math

def single_agent_policy(
    # === Agent Properties ===
    agent_pos: Tuple[float, float],              # Current position of the agent (y, x)
    agent_fire_reduction_power: float,           # How much fire the agent can reduce
    agent_suppressant_num: float,                # Amount of fire suppressant available

    # === Team Information ===
    other_agents_pos: List[Tuple[float, float]], # Positions of all other agents [(y1, x1), (y2, x2), ...]

    # === Fire Task Information ===
    fire_pos: List[Tuple[float, float]],         # Locations of all fires [(y1, x1), (y2, x2), ...]
    fire_levels: List[int],                    # Current intensity level of each fire
    fire_intensities: List[float],               # Current intensity value of each fire task

    # === Task Prioritization ===
    fire_putout_weight: List[float],             # Priority weights for fire suppression tasks
) -> int:
    num_tasks = len(fire_pos)
    
    # Initialize scoring
    task_scores = [0.0] * num_tasks
    
    # Parameters influencing decision making
    distance_temperature = 0.5
    intensity_temperature = 0.4
    weight_temperature = 5.0
    resource_temperature = 0.3
    
    agent_y, agent_x = agent_pos

    # Compute scores for each task
    for i in range(num_tasks):
        fire_y, fire_x = fire_pos[i]
        fire_intensity = fire_intensities[i]
        fire_level = fire_levels[i]
        priority_weight = fire_putout_weight[i]
        
        # Compute distance
        distance = math.sqrt((fire_y - agent_y) ** 2 + (fire_x - agent_x) ** 2)
        
        # Assume a task score based on weight, distance, fire intensity, and available resources
        scaled_distance = math.exp(-distance / distance_temperature)
        scaled_intensity = math.exp(-fire_intensity / intensity_temperature * (1 / (agent_fire_reduction_power + 1e-5)))
        weight_factor = math.exp(priority_weight / weight_temperature)
        resource_factor = max(0, math.log(agent_suppressant_num * agent_fire_reduction_power - fire_intensity + 1) / resource_temperature)
        
        task_score = weight_factor * scaled_distance * scaled_intensity * resource_factor
        
        task_scores[i] = task_score

    # Choose task with highest score
    selected_task_index = max(range(num_tasks), key=lambda index: task_scores[index])

    return selected_task_index