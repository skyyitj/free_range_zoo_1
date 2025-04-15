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
    """
    ... same docstring ...
    """
    task_scores = []

    # Add a small value to avoid division by zero
    epsilon = 1e-8

    # Evaluate each task
    for i in range(len(fire_pos)):
        task_pos = fire_pos[i]
        task_fire_level = fire_levels[i]
        task_fire_intensity = fire_intensities[i]

        # Calculate distance to task
        distance = (abs(agent_pos[0] - task_pos[0]) ** 2 + abs(agent_pos[1] - task_pos[1]) ** 2) ** 0.5

        # Check if other agents are closer to the task
        for other_agent_pos in other_agents_pos:
            other_agent_distance = (abs(other_agent_pos[0] - task_pos[0]) ** 2 + abs(other_agent_pos[1] - task_pos[1]) ** 2) ** 0.5
            if other_agent_distance < distance:
                # Skip this task, as other agents are closer and can handle it more effectively
                break
        else:
            # All other agents are farther away from this task, so consider it
            # Task score is a function of fire level, fire intensity, agent's fire reduction power, and available suppressant
            # Adjust the weights of distance, fire intensity and available suppressant to improve the score function
            task_score = (task_fire_level - distance * 0.7 * task_fire_intensity) * (agent_fire_reduction_power * np.sqrt(agent_suppressant_num + epsilon))
            task_scores.append((i, task_score))

    if not task_scores:
        return -1
    else:
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores[0][0]