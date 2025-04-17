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
import csv

FRZ_ROOT = os.getcwd()
# ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
# ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
import openai

# openai.api_key = "*************************"
# openai.api_base = "https://xiaoai.plus/v1"
openai.api_base = "https://api.lqqq.ltd/v1"
openai.api_key = "sk-ZyDHIVttk1Bkc3Tg75D3129371584a369400868297B82dCa"


def extract_code_from_response(response: str) -> str:
    """从LLM响应中提取第一个python代码块"""
    # print("==========response=============")
    # print(response)
    # print("=======================")
    code_patterns = [
        r'```python(.*?)```',  # 匹配python代码块
        r'```(.*?)```',  # 匹配无语言标注的代码块
        r'python(.*?)',  # 匹配标准python代码
        r'^def\s+.*?(?=\Z)'  # 匹配以def开头直到结尾的代码
    ]

    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            if pattern.startswith('^def'):
                return match.group(0).strip()
            return match.group(1).strip()

    # 没有找到任何代码块时返回空字符串
    return response


@hydra.main(config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    print("workspace_dir")
    print(workspace_dir)
    cache = DiskCache(load_cache=cfg.load_cache)

    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    env_name = cfg.env.env_name.lower()

    env_parent = 'wildfire'
    chunk_size = cfg.sample if "gpt-3.5" in model else 4
    # prompt_dir = f'{FRZ_ROOT}/utils/prompts'
    prompt_dir = f'{FRZ_ROOT}/utils/contrast_prompt'
    task_obs_file = f'{FRZ_ROOT}/envs/{env_parent}/env/{env_name}_obs.py'
    # task_obs_code_string = file_to_string(task_obs_file)
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    # initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    """
    todo: 1. 添加 cfg.threshold 到 initial_system 中
    """
    # initial_user = initial_user.format(threshold=cfg.threshold)
    policy_signature = file_to_string(f'{prompt_dir}/policy_signature.txt')
    # policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    best_policy = file_to_string(f'{prompt_dir}/best_policy.txt')
    worst_policy = file_to_string(f'{prompt_dir}/worst_policy.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
    metric_description = file_to_string(f'{prompt_dir}/metric_description.txt')
    initial_system = initial_system.format(policy_signature=policy_signature, metric_description=metric_description, code_output_tip=code_output_tip)
    # initial_system = initial_system.format(task_obs_code_string=task_obs_code_string, reward_signature=policy_signature)
    # initial_user = initial_user.format(task_obs_code_string=task_obs_code_string,
    #                                    task_description=task_description,
    #                                    metric_description=metric_description
    #                                    )

    # messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    # messages = [{"role": "system", "content": initial_system}]

    GENERATED_AGENT_PATH = f'{FRZ_ROOT}/envs/wildfire/baselines/generated_agent.py'
    GENERATED_AGENT_OLD_PATH = f'{FRZ_ROOT}/envs/wildfire/baselines/generated_agent_old.py'
    target_dir = f"{FRZ_ROOT}/outputs"  # 指定保存目录
    os.makedirs(target_dir, exist_ok=True)  # 自动创建目录
    # 将结果写入 CSV 文件

    csv_path = os.path.join(workspace_dir, 'average_results.csv')
    # 将结果写入 CSV 文件
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Average Fire Intensity Change per step', 'Average Total Rewards', 'Average Suppressant Efficiency'])
    # 全局信息单独定义
    # global_messages = [{"role": "system", "content": initial_system}]
    print(" =================== initial_system =================== ")
    print(initial_system)
    print(" =================== initial_user =================== ")
    print(initial_user)
    global_messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
    # 初始化messages，初始阶段只保留全局信息
    messages = global_messages.copy()
    with open(f"best_sample_idx.txt", "w") as f:
        f.write(f"best_sample_idx\n")
    with open(f"worst_sample_idx.txt", "w") as f:
        f.write(f"worst_sample_idx\n")
    for iter in range(cfg.iteration):
        num1 = 0
        num_success = 0

        responses = []
        total_samples = 0

        while True:
            if total_samples >= cfg.sample:
                break
            kwargs = {
                "mode": model,
                "messages": messages,
                "temperature": cfg.temperature,
            }
            for attempt in range(1000):
                try:
                    if kwargs in cache and cfg.load_cache:
                        print('(using cache)', end=' ')
                        response_cur = cache[kwargs]
                    else:
                        response_cur = openai.ChatCompletion.create(
                            model=model,
                            messages=messages,
                            temperature=cfg.temperature,
                            n=chunk_size
                        )
                    cache[kwargs] = response_cur
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= cfg.attempt_times:
                        chunk_size = max(int(chunk_size / 2), 1)
                        # print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt + 1} failed with error: {e}")
                    time.sleep(1)

            responses.extend(response_cur["choices"])

        agent_files = []

        for response_id in range(cfg.sample):
            num1 = num1 + 1
            response_cur = responses[response_id]["message"]["content"]
            code_string = extract_code_from_response(response_cur)
            functions_code = code_string.strip()
            # 将函数保存为单独的python文件
            function_filename = f"agent_iter{iter}_response{response_id}.py"
            with open(function_filename, 'w') as f:
                f.write(functions_code)

            if not os.path.exists("original_response"):
                os.makedirs("original_response")

            response_filename = f"original_response/agent_iter{iter}_response{response_id}.txt"
            with open(response_filename, 'w') as f:
                f.write(response_cur)
            agent_files.append(function_filename)
            # 更新generate_agent.py文件
            # 先将原始文件内容复制到目标文件
            with open(GENERATED_AGENT_OLD_PATH, 'r') as old_file:
                old_content = old_file.read()
            # 将原始内容写入目标文件
            with open(GENERATED_AGENT_PATH, 'w') as file:
                file.write(old_content)
            with open(GENERATED_AGENT_PATH, 'a') as file:
                file.write('\n\n' + functions_code)
            # 读取文件内容
            with open(GENERATED_AGENT_PATH, 'r') as file:
                lines = file.readlines()
            # 查找并替换第一次出现"class"的行
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    lines[i] = 'class GenerateAgent(Agent):\n'
                    break
            # 将更新后的内容写回文件
            with open(GENERATED_AGENT_PATH, 'w') as file:
                file.writelines(lines)
            rl_filepath = f"agent_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen([
                    'python', f'{FRZ_ROOT}/experiments/train.py',
                    '--agent', 'generated_agent',
                    '--env', f'{env_parent}{suffix.lower()}',
                    '--config', 'default',
                    '--seed', str(42 + response_id)
                ], stdout=f, stderr=f)
            process.wait()
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)

        # 后续结果分析与进化选择逻辑

        best_agent_code = ""
        worst_agent_code = ""

        # best_score, best_agent = -float('inf'), None

        baseline_dir = f"{FRZ_ROOT}/envs/wildfire/baselines"
        os.makedirs(baseline_dir, exist_ok=True)
        all_rewards = []
        all_contents = []
        for response_id in range(len(agent_files)):
            rl_filepath = f"agent_iter{iter}_response{response_id}.txt"

            with open(rl_filepath, 'r') as f:
                stdout_str = f.read()

            traceback_msg = filter_traceback(stdout_str)  # 查找是否有traceback信息
            content = ''
            if traceback_msg == '':
                num_success = num_success + 1
                # 替换tensorboard相关代码
                # content += policy_feedback.format(metric_description=metric_description)

                # 读取rl_filepath中的内容
                with open(rl_filepath, 'r') as f:
                    lines = f.readlines()
                final_rewards = 0
                for line in lines:
                    if "Average" in line and "Rewards:" in line:
                        # 保存上一个step的reward总和
                        reward_str = line.split("Average Rewards:")[-1].strip()
                        reward = float(reward_str)
                        final_rewards = reward

                with open(rl_filepath, 'r') as file:
                    lines = file.readlines()
                # 标记是否开始记录内容
                recording = False

                # 遍历文件的每一行
                for line in lines:
                    # 检查是否到达 "Step 1 rewards:" 行
                    if "Average Evaluate Metrics:" in line:
                        recording = True

                    # 如果开始记录，将当前行加入 content
                    if recording:
                        content += line

                    # 检查是否到达 "training process finished!" 行
                    if "training process finished!" in line:
                        break

                content += code_feedback
            else:
                final_rewards = -float('inf')
                content += execution_error_feedback.format(traceback_msg=traceback_msg)
            all_rewards.append(final_rewards)
            all_contents.append(content)

        best_sample_idx = np.argmax(np.array(all_rewards))
        worst_sample_idx = np.argmin(np.array(all_rewards))
        with open(f"best_sample_idx.txt", "a") as f:
            f.write(f"best_sample_idx: {best_sample_idx}\n")
        txt1 = best_policy.format(metric_description=metric_description)
        best_content = txt1 + all_contents[best_sample_idx]
        best_agent_code = responses[best_sample_idx]["message"]["content"]

        with open(f"worst_sample_idx.txt", "a") as f:
            f.write(f"worst_sample_idx: {worst_sample_idx}\n")
        txt2 = worst_policy
        worst_content = txt2 + all_contents[worst_sample_idx]
        worst_agent_code = responses[worst_sample_idx]["message"]["content"]
        # contents += content
        # contents += f"The success rate for this iteration is {num_success}/{num1}\n"

        # if best_agent:
        #     logging.info(f"Best agent in iteration {iter} is {best_agent} with score {best_score}")
        #     best_agent_baseline = os.path.join(baseline_dir, os.path.basename(best_agent))
        #     shutil.copy(best_agent, best_agent_baseline)

        #     with open(best_agent, 'r') as file:
        #         best_agent_code = file.read()
        #     # messages.append({
        #     #     "role": "assistant",
        #     #     "content": best_agent_code
        #     # })
        #     feedback = f"The best agent achieved a score of {best_score}. Based on this, refine the agent further."
        #     # messages.append({"role": "user", "content": feedback})

        # else:
        #     feedback = "None of the agents performed well. Please propose a significantly different approach."
        #     # messages.append({"role": "user", "content": feedback})

        # # 完整反馈信息追加到 messages
        # for content in contents:
        #     messages.append({"role": "user", "content": content})
        # print("messages:", messages)
        # iteration_messages = prepare_iteration_messages(best_agent_code, feedback, contents)
        iteration_messages1 = [{"role": "assistant", "content": best_agent_code},
                               {"role": "user", "content": best_content}]

        iteration_messages2 = [{"role": "assistant", "content": worst_agent_code},
                               {"role": "user", "content": worst_content}]
        messages = global_messages + iteration_messages1 + iteration_messages2
        #print("messages:", messages)


if __name__ == "__main__":
    main()
