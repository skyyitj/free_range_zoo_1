import numpy as np

def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float, 
    agent_suppressant_num: float,

    other_agents_pos: List[Tuple[float, float]],

    fire_pos: List[Tuple[float, float]],
    fire_levels: List[int],
    fire_intensities: List[float],

    fire_putout_weight: List[float],
) -> int:
    num_tasks = len(fire_pos)
    scores = []

    suppressant_sufficiency_score_temperature = 0.1
    fire_intensity_score_temperature = 0.1

    for task_index in range(num_tasks):
        fire_y, fire_x = fire_pos[task_index]
        fire_level = fire_levels[task_index]
        fire_intensity = fire_intensities[task_index]
        weight = fire_putout_weight[task_index]

        dist = np.sqrt((agent_pos[0] - fire_y)**2 + (agent_pos[1] - fire_x)**2)

        # Calculate how effectively the agent can reduce this particular fire
        potential_suppressant_use = min(agent_suppressant_num, fire_intensity / agent_fire_reduction_power)
        suppressant_sufficiency_score = potential_suppressant_use / (fire_intensity / agent_fire_reduction_power)

        # Normalize values to ensure balanced contribution across scores
        normalized_suppressant_sufficiency_score = np.exp(suppressant_sufficiency_score_temperature * suppressant_sufficiency_score)
        normalized_fire_intensity_score = np.exp(-fire_intensity_score_temperature * fire_intensity)
        
        # Composite score for this task
        score = weight * normalized_suppressant_sufficiency_score * normalized_fire_intensity_score / (dist + 1)
        scores.append(score)
    
    # Choose task with highest composite score
    return int(np.argmax(scores))