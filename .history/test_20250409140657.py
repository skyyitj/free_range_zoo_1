import re

# @hydra.main(config_path="cfg", config_name="config")
def main():
    response_cur="Here's the policy function for the agent. It will select the task of fire extinguishing based on an evaluation of all fires' intensity and level balanced with the position distance of the agent and other agents. It will also ensure that it has enough suppressant and retreats when resources are depleted.\n\n```python\n# Implied helper functions:\nfrom typing import Tuple, List\n\ndef calculate_moves(agent_pos: Tuple[float, float], task_pos: Tuple[float, float]) -> float:\n    return ((task_pos[0] - agent_pos[0]) ** 2 + (task_pos[1] - agent_pos[1]) ** 2) ** 0.5\n\ndef calculate_task_power(fire_level: float, fire_intensity: float) -> float:\n    return fire_level * fire_intensity\n\ndef single_agent_policy(\n    agent_pos: Tuple[float, float],\n    agent_fire_reduction_power: float,\n    agent_suppressant_num: float,\n    other_agents_pos: List[Tuple[float, float]],\n    fire_pos: List[Tuple[float, float]],\n    fire_levels: List[float],\n    fire_intensities: List[float],\n    valid_action_space: List[List[int]]\n) -> int:\n    \n    # Initialize maximum task power and chosen task index\n    max_task_power = -1\n    chosen_task_index = -1\n    \n    # Loop through each fire task\n    for task_index, (task_pos, fire_level, fire_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):\n        \n        # calculate the distance from agent to the task\n        moves_to_task = calculate_moves(agent_pos, task_pos)\n        \n        # check if agent has enough suppressant to complete the task\n        if moves_to_task + 1 > agent_suppressant_num:\n            continue\n            \n        # calculate the task power of the fire task based on its level and intensity\n        task_power = calculate_task_power(fire_level, fire_intensity)\n        \n        # Add up the distances of all other agents to the task for collaborative firefighting\n        for other_agent_pos in other_agents_pos:\n            moves_to_task += calculate_moves(other_agent_pos, task_pos)\n            \n        # Balance the evaluation with a division to avoid overly patrolling in low intensity fire. The lower the value, the better\n        evaluation = moves_to_task / task_power\n        \n        # check if this task's evaluation is the lowest\n        if evaluation < max_task_power or max_task_power == -1:\n            max_task_power = evaluation\n            chosen_task_index = task_index\n    \n    return chosen_task_index\n```\nPlease note that you will need to maintain agent states through an environment which is not in the scope of this answer. If there is no valid action for the agent (all suppressant are depleted or all fire tasks are done), then the agent should retreat. This state should be checked on another function that calls this single_agent_policy function."
    
    # 提取所有函数定义
    functions = re.findall(
            r'(?ms)(^def\s+\w+\(.*?\)(?:\s*->\s*.+)?:\n(?:\s+.*\n)+)',
            response_cur,)
    functions_code = '\n'.join(functions)

    functions_code = functions_code.rstrip()  # 先去除尾部空白字符
    
    if functions_code.endswith("'''") or functions_code.endswith('"""'):
        functions_code = functions_code[:-3].rstrip()

    print("function_code:",functions_code)
            

if __name__ == "__main__":
    main()