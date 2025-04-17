import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time
from utils.misc import *
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from LLM_cache import DiskCache
FRZ_ROOT = os.getcwd()
# ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
# ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
import openai
# openai.api_key = "*************************"
# openai.api_base = "https://xiaoai.plus/v1"
openai.api_key = "********************"
openai.api_base = "https://api.lqqq.ltd/v1"
# openai.api_key = "*************************"
@hydra.main(config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    workspace_dir = workspace_dir / "../../../"
    print(workspace_dir)
    exit(0)
    cache = DiskCache(load_cache=cfg.load_cache)

    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    env_name = cfg.env.env_name.lower()

    env_parent = 'wildfire'

    task_obs_file = f'{FRZ_ROOT}/envs/{env_parent}/env/{env_name}_obs.py'
    example_code = f'{FRZ_ROOT}/envs/{env_parent}/env/example_code.py'
    task_obs_code_string = file_to_string(task_obs_file)
    example_code_string = file_to_string(example_code)
    # prompt_dir = f'{FRZ_ROOT}/utils/prompts'
    prompt_dir = f'{FRZ_ROOT}/utils/revised_prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    # code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    # code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    # initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    policy_signature = file_to_string(f'{prompt_dir}/policy_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    # initial_system = initial_system.format(reward_signature=reward_signature) + code_output_tip
    initial_system = initial_system.format(policy_signature=policy_signature)
    # initial_user = initial_user.format(task_obs_code_string=task_obs_code_string,
    #                                    task_description=task_description)
    # messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    messages = [{"role": "system", "content": initial_system}]

    for iter in range(cfg.iteration):

        # total_samples = 0
        # chunk_size = cfg.sample if "gpt-3.5" in model else 4

        rl_runs = []
        responses = []
        for _ in range(cfg.sample):
            # 直接请求API，不再检查缓存
            response_cur = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=cfg.temperature,
                n=1
            )
            responses.append(response_cur["choices"][0])  # 直接添加最新响应
        print("response:")
        print(response_cur)
        successes, reward_correlations, agent_files = [], [], []

        # for response_id in range(cfg.sample):
        #     response_cur = responses[response_id]["message"]["content"]
        #
        #     patterns = [r'```python(.*?)```', r'```(.*?)```']
        #     code_string = None
        #     for pattern in patterns:
        #         match = re.search(pattern, response_cur, re.DOTALL)
        #         if match:
        #             code_string = match.group(1).strip()
        #             break
        #     if not code_string:
        #         code_string = response_cur.strip()
        #
        #     lines = code_string.split("\n")
        #     for i, line in enumerate(lines):
        #         if line.strip().startswith("class "):
        #             code_string = "\n".join(lines[i:])
        #             break

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]

            # 匹配所有python代码块
            pattern = r'```python(.*?)```|```(.*?)```'
            matches = re.findall(pattern, response_cur, re.DOTALL)

            response_code_strings = []  # 存储当前response的所有代码块

            for match in matches:
                code_string = match[0] if match[0] else match[1]
                code_string = code_string.strip()

                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("class "):
                        extracted_code = "\n".join(lines[i:])
                        response_code_strings.append(extracted_code)
                        break

            # 单个response内的所有代码块组合
            combined_response_code = "\n\n".join(response_code_strings)

            agent_filename = f"agent_iter{iter}_response{response_id}.py"
            with open(agent_filename, 'w') as file:

                file.write("# Auto-generated Agent Class\n")
                file.write("from typing import Tuple, List, Dict, Any\n")
                file.write("import torch\n")
                file.write("import free_range_rust\n")
                file.write("from torch import Tensor\n")
                file.write("from free_range_rust import Space\n")
                file.write("import numpy as np\n")
                file.write("from collections import defaultdict\n")
                file.write("from free_range_zoo.utils.agent import Agent\n")

                file.write(combined_response_code + '\n')

            agent_files.append(agent_filename)
            shutil.copy(agent_filename,
                        '/home/liuchi/yitianjiao/aamas2025/free-range-zoo/free_range_zoo/envs/wildfire/baselines/generated_agent.py')

            # 打开目标文件并替换第一次出现的 'class' 行
            with open(
                    '/home/liuchi/yitianjiao/aamas2025/free-range-zoo/free_range_zoo/envs/wildfire/baselines/generated_agent.py',
                    'r+') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith('class '):
                        lines[i] = 'class GenerateAgent(Agent):\n'
                        break
                file.seek(0)
                file.writelines(lines)
                file.truncate()

            # 添加等待时间以避免并行错误
            time.sleep(0.5)

            print("Current working directory:", os.getcwd())
            rl_filepath = f"agent_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen([
                    'python', f'{FRZ_ROOT}/experiments/train.py',
                    '--agent', agent_filename,
                    '--env', f'{env_parent}{suffix.lower()}',
                    '--config', 'default',
                    '--seed', '42'
                ], stdout=f, stderr=f)
            process.wait()
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)


        # 后续结果分析与进化选择逻辑
        best_score, best_agent = -float('inf'), None
        contents = []

        baseline_dir = f"{FRZ_ROOT}/envs/wildfire/baselines"
        os.makedirs(baseline_dir, exist_ok=True)

        for response_id, agent_file in enumerate(agent_files):
            log_filepath = rl_filepath

            with open(log_filepath, 'r') as f:
                stdout_str = f.read()

            traceback_msg = filter_traceback(stdout_str)  # 查找是否有traceback信息
            content = ''

            if traceback_msg == '':
                # 替换tensorboard相关代码
                content += policy_feedback.format(epoch_freq=3)

                # 读取rl_filepath中的内容
                with open(log_filepath, 'r') as f:
                    lines = f.readlines()

                # 提取step rewards
                # 提取step rewards
                step_rewards = []
                current_step_reward = 0
                final_rewards = []
                is_final_section = False
                for line in lines:
                    if "Step" in line and "rewards:" in line:
                        # 保存上一个step的reward总和
                        if current_step_reward > 0:  # 确保不是第一次
                            step_rewards.append(current_step_reward)
                        current_step_reward = 0  # 重置当前step的reward
                    elif is_final_section == False and "Agent" in line and "total reward:" in line:
                        # 处理tensor格式的reward
                        reward_str = line.split("total reward:")[-1].strip()
                        # 移除tensor([])格式，只保留数值
                        reward = float(reward_str.replace("tensor([", "").replace("])", ""))
                        current_step_reward += reward
                    elif "Final total rewards:" in line:
                        step_rewards.append(current_step_reward)
                        is_final_section = True
                    elif is_final_section and "Agent" in line and "total reward:" in line:
                        reward_str = line.split("total reward:")[-1].strip()
                        # 同样处理tensor格式
                        reward = float(reward_str.replace("tensor([", "").replace("])", ""))
                        final_rewards.append(reward)

                    # 添加step rewards到content
                if len(step_rewards) > 2:
                    content += "\nStep rewards:\n"
                    for i, reward in enumerate(step_rewards):  # 每三个step取一个
                        content += f"Step {(i + 1) * 3} total reward: {reward:.2f}\n"

                    # 添加final rewards到content
                if final_rewards:
                    content += "\nFinal rewards:\n"
                    total_final_reward = sum(final_rewards)
                    content += f"Total final reward: {total_final_reward:.2f}\n"

                content += code_feedback

                # 计算successes
                if final_rewards:
                    # 使用final rewards的总和作为max_reward
                    max_reward = sum(final_rewards)
                    successes.append(max_reward)

                    if max_reward > best_score:
                        best_score = max_reward
                        best_agent = agent_file
                else:
                    successes.append(-float('inf'))
            else:
                successes.append(-float('inf'))
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        if best_agent:
            logging.info(f"Best agent in iteration {iter} is {best_agent} with score {best_score}")
            best_agent_baseline = os.path.join(baseline_dir, os.path.basename(best_agent))
            shutil.copy(best_agent, best_agent_baseline)

            with open(best_agent, 'r') as file:
                best_agent_code = file.read()

            messages.append({
                "role": "assistant",
                "content": best_agent_code
            })
            feedback = f"The best agent achieved a score of {best_score}. Based on this, refine the agent further."
            messages.append({"role": "user", "content": feedback})
        else:
            feedback = "None of the agents performed well. Please propose a significantly different approach."
            messages.append({"role": "user", "content": feedback})

        # 完整反馈信息追加到 messages
        for content in contents:
            messages.append({"role": "user", "content": content})
        print("messages:")
        print(messages)


if __name__ == "__main__":
    main()
