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
    best_task_score = float('-inf')
    selected_task_index = -1

    # Temperature adjustments for policy tuning
    distance_temperature = 0.009  # Lower to allow more flexibility
    intensity_temperature = 0.017  # Lower slightly to handle intense fires more strategically
    suppressant_conserve_factor = 11.0  # Even more emphasis on saving suppressant
    reward_weight_focus = 3.0  # Increased to focus more on higher-reward tasks
    burnout_penalty = 5.0  # Introduce a penalty for enabling fire burnouts

    # Normalize the suppressant factor: encourage saving suppressant for situations with more intense fires
    remaining_suppressant = np.exp(-suppressant_conserve_factor * (1 - (agent_suppressant_num / 10)))

    for i in range(num_tasks):
        distance = np.sqrt((agent_pos[0] - fire_pos[i][0])**2 + (agent_pos[1] - fire_pos[i][1])**2)
        norm_distance = np.exp(-distance_temperature * distance)

        intensity = fire_intensities[i] * fire_levels[i]
        norm_intensity = np.exp(-intensity_temperature * intensity)

        # Reward weight enhancement: larger weight leads to more focus on high-reward tasks
        reward_focus = np.exp(reward_weight_focus * fire_putout_weight[i])
        
        # Burnout penalty: larger penalty for fires too intense for an agent to handle
        fire_burnout_penalty = np.exp(-burnout_penalty * (intensity / agent_fire_reduction_power))

        # Adjust score to include higher priorities on larger reward tasks and avoid fire burnouts
        score = reward_focus * (norm_distance * agent_fire_reduction_power / (1 + intensity)) * np.log1p(fire_levels[i]) * remaining_suppressant * norm_intensity * fire_burnout_penalty

        if score > best_task_score:
            best_task_score = score
            selected_task_index = i

    return selected_task_index