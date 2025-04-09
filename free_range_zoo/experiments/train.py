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
import os
import numpy as np

workspace_dir = Path.cwd()
print('workspace_dir:', workspace_dir)
with open(os.path.join(workspace_dir, "../../archive/competition_configs/wildfire/WS1.pkl"), "rb") as file:
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

agents = {agent_name: GenerateAgent(env.action_space(agent_name), parallel_envs=1, configuration=wildfire_configuration)
          for agent_name in env.agents}

total_rewards = {agent_name: 0.0 for agent_name in env.agents}  # 记录每个智能体的总奖励
step_counter = 0  # 添加step计数器

# 添加记录火焰强度变化的列表
fire_intensity_changes = []

# 记录灭火效率相关数据
efficiency_records = []
extinguished_total = 0

while not torch.all(env.finished):
    step_counter += 1  # 增加step计数
    # 添加灭火效率相关的变量
    total_suppressant_before = 0
    for agent_name, observation in observations.items():
        obs, t_mapping = observation
        # 获取智能体的灭火剂量 - obs['self'] 形状为 tensor([[0., 0., 1., 2.]])
        agent_suppressant = obs['self'][0, 3].item()  # 获取索引为[0,3]的元素
        total_suppressant_before += agent_suppressant
    # 计算火焰强度总和
    total_fire_intensity_before = 0
    total_fire_intensity_after = 0
    total_fire_count_before = 0  # 记录火焰数量

    # 解包 observation 并提取 tasks 中的火焰信息
    for agent_name, observation in observations.items():
        obs, t_mapping = observation
        # 处理嵌套张量
        tasks_nested = obs['tasks']
        # 从嵌套张量中提取普通张量
        for tensor_item in tasks_nested.unbind():
            # 获取火焰数量（即行数）
            fire_count = tensor_item.shape[0]
            total_fire_count_before += fire_count
            # 提取第三列（火焰强度）
            fire_intensities = tensor_item[:, 3]
            total_fire_intensity_before += fire_intensities.sum().item()
        break

    # 继续执行智能体的动作和环境的更新
    for agent_name, agent in agents.items():
        observation = observations[agent_name]
        if isinstance(observation, tuple):
            agent.observe(observation)
        else:
            agent.observe((observation, None))

    agent_actions = {
        agent_name: agents[agent_name].act(action_space=env.action_space(agent_name))
        for agent_name in env.agents
    }

    next_observations, rewards, terminations, truncations, infos = env.step(agent_actions)

    # 每个step输出一次reward
    print(f"\nStep {step_counter} rewards:")
    for agent_name, reward in rewards.items():
        total_rewards[agent_name] += reward
        print(f"Agent {agent_name} ,total reward: {total_rewards[agent_name]}")

    # 计算火焰强度总和和火焰数量（之后）
    total_fire_count_after = 0  # 记录火焰数量
    for agent_name, observation in next_observations.items():
        obs, t_mapping = observation
        # 处理嵌套张量
        tasks_nested = obs['tasks']
        # 从嵌套张量中提取普通张量
        for tensor_item in tasks_nested.unbind():
            # 获取火焰数量（即行数）
            fire_count = tensor_item.shape[0]
            total_fire_count_after += fire_count
            # 提取第三列（火焰强度）
            fire_intensities = tensor_item[:, 3]
            total_fire_intensity_after += fire_intensities.sum().item()
        break

    # 计算火焰变化
    fire_intensity_change = total_fire_intensity_after - total_fire_intensity_before
    fire_count_change = total_fire_count_after - total_fire_count_before

    # 记录火焰强度变化
    fire_intensity_changes.append(fire_intensity_change)

    # 添加灭火效率计算
    total_suppressant_after = 0
    for agent_name, observation in next_observations.items():
        obs, t_mapping = observation
        agent_suppressant = obs['self'][0, 3].item()  # 获取索引为[0,3]的元素
        total_suppressant_after += agent_suppressant

    # 计算灭火剂使用量
    suppressant_used = max(0, total_suppressant_before - total_suppressant_after)

    # 计算灭火效率
    fire_intensity_reduction = max(0, total_fire_intensity_before - total_fire_intensity_after)  # 火焰强度的减少量
    extinguish_efficiency = 0
    if suppressant_used > 0 and fire_intensity_reduction > 0:
        extinguish_efficiency = fire_intensity_reduction / suppressant_used
        efficiency_records.append(extinguish_efficiency)

    extinguished_fires = max(0, -fire_count_change)  # 只考虑减少的火点(即扑灭的火点)
    extinguished_total += extinguished_fires
    # 更新下一步的初始灭火剂量
    total_suppressant_before = total_suppressant_after

    # 输出本次步骤的火焰强度变化
    print(
        f"  - Fire Intensity Change: {fire_intensity_change:.2f} ({'increase' if fire_intensity_change > 0 else 'decrease' if fire_intensity_change < 0 else 'no change'})")
    # 输出灭火效率指标
    print(
        f"  - Fire Count Change: {fire_count_change} ({'increase' if fire_count_change > 0 else 'decrease' if fire_count_change < 0 else 'no change'})")
    if suppressant_used > 0:
        print(f"  - Suppressant Used: {suppressant_used:.2f}")
    if fire_intensity_reduction > 0:
        print(f"  - Fire Intensity Reduction: {fire_intensity_reduction:.2f}")
    if extinguish_efficiency > 0:
        print(f"  - Extinguish Efficiency: {extinguish_efficiency:.4f} intensity/suppressant")
    if extinguished_fires > 0:
        print(f"  - Fires Extinguished: {extinguished_fires}")

    # 更新观察值
    observations = next_observations

env.close()

# 打印最终的总奖励和平均每步火焰强度变化
print("\nFinal total rewards:")
for agent_name, reward in rewards.items():
    print(f"Agent {agent_name} ,total reward: {total_rewards[agent_name]}")
# 计算并输出平均每步火焰强度变化
if fire_intensity_changes:
    avg_intensity_change = np.mean(fire_intensity_changes)
    print(f"\nFire Intensity Change Statistics:")
    print(f"  - Average fire intensity change per step: {avg_intensity_change:.2f}")
    print(f"  - Total fire intensity change: {sum(fire_intensity_changes):.2f}")
    print(f"  - Steps with fire intensity increase: {sum(1 for change in fire_intensity_changes if change > 0)}")
    print(f"  - Steps with fire intensity decrease: {sum(1 for change in fire_intensity_changes if change < 0)}")
    print(f"  - Steps with stable fire intensity: {sum(1 for change in fire_intensity_changes if change == 0)}")

# 输出灭火效率的总体统计
print("\nExtinguish Efficiency Statistics:")
if efficiency_records:
    avg_efficiency = sum(efficiency_records) / len(efficiency_records)
    max_efficiency = max(efficiency_records)
    print(f"  - Total Fires Extinguished: {extinguished_total}")
    print(f"  - Average Extinguish Efficiency: {avg_efficiency:.4f} intensity/suppressant")
    print(f"  - Maximum Extinguish Efficiency: {max_efficiency:.4f} intensity/suppressant")
    print(f"  - Steps with Effective Extinguishing: {len(efficiency_records)}")
else:
    print("  - No effective fire extinguishing occurred")

print("training process finished!")
