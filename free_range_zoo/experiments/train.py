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
import random

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
import csv
from datetime import datetime
import inspect
from free_range_zoo.envs.wildfire.baselines import GenerateAgent,StrongestBaseline

workspace_dir = Path.cwd()
print('workspace_dir:', workspace_dir)
total_metrics = dict()
average_fire_intensity_change = 0
average_suppressant_efficiency = 0

for k in range(1,4):

    with open(os.path.join(f"/home/liuchi/yitianjiao/aamas2025/free-range-zoo/archive/competition_configs/wildfire/WS{k}.pkl"), "rb") as file:
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


    agents = {agent_name: GenerateAgent(env.action_space(agent_name), parallel_envs=1, configuration=wildfire_configuration)
              for agent_name in env.agents}
    #agents = {agent_name: StrongestBaseline(env.action_space(agent_name), parallel_envs=1)
    #           for agent_name in env.agents}
    total_rewards = {agent_name: 0.0 for agent_name in env.agents}  # 记录每个智能体的总奖励
    step_counter = 0  # 添加step计数器

    # 添加记录火焰强度变化的列表
    fire_intensity_changes = []

    # 记录灭火效率相关数据
    efficiency_records = []
    extinguished_total = 0

    # 初始化总计数器
    total_fire_intensity_changes = []
    total_rewards_list = []
    total_extinguished_fires = []
    results = []
    all_metrics = dict()

    # 进行十次测试
    for test_run in range(10):
        # 重置环境和智能体
        env.reset()
        step_counter = 0
        observations, infos = env.reset()
        total_rewards = {agent_name: 0.0 for agent_name in env.agents}
        fire_intensity_changes = []
        extinguished_total = 0

        metrics = dict()
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
            for key in infos['rewards'].keys():

                if key not in metrics.keys():
                    metrics[key] = infos['rewards'][key]
                else:
                    metrics[key] += infos['rewards'][key]



            # 每个step输出一次reward
            #print(f"\nStep {step_counter} rewards:")
            for agent_name, reward in rewards.items():
                total_rewards[agent_name] += reward
                #print(f"Agent {agent_name} ,total reward: {total_rewards[agent_name]}")

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
            if "Fire Intensity Change" not in metrics.keys():
                metrics["Fire Intensity Change"] = fire_intensity_change
            else:
                metrics["Fire Intensity Change"] += fire_intensity_change
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
            if "Used Suppressant" not in metrics.keys():
                metrics["Used Suppressant"] = suppressant_used
            else:
                metrics["Used Suppressant"] += suppressant_used


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

            # 更新观察值
            observations = next_observations

        print('test_run:', test_run, 'step_counter:', step_counter)
        if 'Steps' not in metrics.keys():
            metrics['Steps'] = [step_counter]
        else:
            metrics['Steps'].append(step_counter)

        metrics['Burning Number'] /= step_counter
        # 记录每次测试的结果
        total_fire_intensity_changes.append(sum(fire_intensity_changes))
        if 'Rewards' not in metrics.keys():
            metrics['Rewards'] = [sum(total_rewards.values())]
        else:
            metrics['Rewards'].append(sum(total_rewards.values()))
        # total_rewards_list.append(sum(total_rewards.values()))
        total_extinguished_fires.append(extinguished_total)

        for key in metrics.keys():
            if key not in all_metrics.keys():
                all_metrics[key] = [metrics[key]]
            else:
                all_metrics[key].append(metrics[key])
        # 输出每次测试的结果
        # print(f"Test {test_run + 1} Metric:")
        # print(f"  - Total Rewards: {total_rewards_list[-1].item():.2f}")
        # for key in metrics.keys():
        #     print(f"  - {key}: {metrics[key]:.2f}")
        # print(f"  - Fire Intensity Change per step: {total_fire_intensity_changes[-1] / step_counter:.2f}")
        results.append(total_fire_intensity_changes[-1] / step_counter)

        # print(f"  - Suppressant Efficiency: {extinguish_efficiency:.4f} intensity/suppressant")



    # 计算并输出十次测试的平均值

    average_fire_intensity_change += np.mean(results)
    # average_total_rewards = np.mean(total_rewards_list)
    average_suppressant_efficiency += np.mean(efficiency_records)
    # average_used_suppressant_number = np.mean(total_suppressant_before)
    print("\nAverage Evaluate Metrics:")

    # print(f"  - Average Rewards: {average_total_rewards:.2f}")
    for key in all_metrics.keys():
        # print(f"  - Average {key}: {np.mean(all_metrics[key]):.2f}")
        if key not in total_metrics.keys():
            total_metrics[key] = np.mean(all_metrics[key])
        else:
            total_metrics[key] += np.mean(all_metrics[key])
    env.close()


for key in total_metrics.keys():
    print(f"  - Average {key}: {total_metrics[key]/3:.4f}")
print(f"  - Average Fire Intensity Change: {average_fire_intensity_change/3:.2f}")
    # print(f"  - Average Used Suppressant Number: {average_used_suppressant_number:.2f}")
print(f"  - Average Suppressant Efficiency: {average_suppressant_efficiency/3:.4f} intensity/suppressant")
# 将结果写入 CSV 文件
csv_path = "average_results.csv"

# 检查文件是否存在以及是否为空
file_exists = os.path.isfile(csv_path)
file_is_empty = os.path.getsize(csv_path) == 0 if file_exists else True
keys_list = list(all_metrics.keys()) + ['Average_fire_intensity_change', 'Average Suppressant Efficiency']
values_list = list(all_metrics.values()) + [average_fire_intensity_change, average_suppressant_efficiency]

# 将结果写入 CSV 文件
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    if not file_exists or file_is_empty:
        writer.writerow(keys_list)

    writer.writerow(values_list)


