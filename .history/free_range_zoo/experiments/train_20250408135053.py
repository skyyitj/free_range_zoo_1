# import sys
#
# # 查找并清除所有与 free_range_zoo 相关的模块
# modules_to_delete = [name for name in sys.modules if name.startswith('free_range_zoo')]
#
# for module_name in modules_to_delete:
#     del sys.modules[module_name]
#     print(f"已清除模块缓存: {module_name}")
#
# print("所有与 free_range_zoo 相关的模块缓存已清除。")

# import os
#
# # 设置 PYTHONPATH，添加新的路径
# os.environ['PYTHONPATH'] = '/home/liuchi/yitianjiao/aamas2025/free-range-zoo'
#
# # 验证设置
# print(f"新的 PYTHONPATH: {os.environ.get('PYTHONPATH')}")

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
import torch
import pickle

# import sys
#
# # 查找所有与 free_range_zoo 相关的模块
# free_range_zoo_modules = {name: module for name, module in sys.modules.items() if name.startswith('free_range_zoo')}
#
# # 打印找到的模块
# if free_range_zoo_modules:
#     print(f"找到 {len(free_range_zoo_modules)} 个与 free_range_zoo 相关的模块:")
#     for name in sorted(free_range_zoo_modules.keys()):
#         print(f"- {name}")
# else:
#     print("没有找到与 free_range_zoo 相关的模块。")
# 打开文件并使用 pickle.load 读取配置

from pathlib import Path


workspace_dir = Path.cwd()
print(workspace_dir)
with open("free-range-zoo/archive/competition_configs/wildfire/WS1.pkl"
        , "rb") as file:
    wildfire_configuration = pickle.load(file)

env = wildfire_v0.parallel_env(
    max_steps=100,
    parallel_envs=1,
    override_initialization_check=True,
    configuration=wildfire_configuration,
    device=torch.device('cpu'),
    log_directory="./trainlog_outputs/"
)

env.reset()
env = action_mapping_wrapper_v0(env)
observations, infos = env.reset()
import inspect
from free_range_zoo.envs.wildfire.baselines import GenerateAgent

agents = {agent_name: GenerateAgent(env.action_space(agent_name), parallel_envs=1) for agent_name in env.agents}

total_rewards = {agent_name: 0.0 for agent_name in env.agents}  # 记录每个智能体的总奖励
step_counter = 0  # 添加step计数器

while not torch.all(env.finished):
    step_counter += 1  # 增加step计数

    for agent_name, agent in agents.items():
        # 修改观察值的处理方式
        observation = observations[agent_name]
        if isinstance(observation, tuple):
            agent.observe(observation)
        else:
            agent.observe((observation, None))  # 如果没有映射，传入None

    agent_actions = {
        agent_name: agents[agent_name].act(action_space=env.action_space(agent_name))
        for agent_name in env.agents
    }  # Policy action determination here

    # 执行动作并获取反馈
    next_observations, rewards, terminations, truncations, infos = env.step(agent_actions)

    # 每三个step输出一次reward
    if step_counter % 3 == 0:
        print(f"\nStep {step_counter} rewards:")
        for agent_name, reward in rewards.items():
            total_rewards[agent_name] += reward
            print(f"Agent {agent_name} ,total reward: {total_rewards[agent_name]}")
    else:
        # 不输出时也要累加reward
        for agent_name, reward in rewards.items():
            total_rewards[agent_name] += reward

    observations = next_observations

env.close()

# 打印最终的总奖励
print("\nFinal total rewards:")
for agent_name, reward in rewards.items():
    print(f"Agent {agent_name} ,total reward: {total_rewards[agent_name]}")

print("ok!")
