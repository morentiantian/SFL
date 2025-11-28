# 文件: train.py (HAC-SFL 改进版)
# 修改日期: 2025-06-21
# 状态: 逻辑不变，主要确保参数能正确传递。
# 备注: 此文件负责启动训练流程。我们在此假设所有新的超参数
# (如 mu_initial, divergence_target, w_perf 等)
# 都已在 config.py 的 args_parser() 函数中定义。

import copy
import logging
import os
import sys
import setproctitle
import numpy as np
from pathlib import Path
import torch
import multiprocessing as mp

# --- 路径设置，保持不变 ---
# 1. 获取当前脚本文件(train.py)的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取当前脚本所在的目录 (e.g., /path/to/light_mappo/train)
script_dir = os.path.dirname(current_script_path)
# 3. 获取项目根目录 (e.g., /path/to/light_mappo)
project_root = os.path.dirname(script_dir)
# 4. 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 导入自定义模块，保持不变 ---
from envs.env_wrappers import DummyVecEnv, SubprocVecEnv
# 注意: 这里的导入路径可能需要根据您的项目结构调整
# 如果 envs.env_core 报错，请尝试 from env_core import EnvCore
from envs.env_core import EnvCore 
from config import args_parser

def make_train_env(all_args):
    # 创建训练环境，保持不变# 
    def get_env_fn(rank):
        def init_env():
            # 为每个环境设置独立的日志目录
            env_log_dir = Path(all_args.log_dir) / "envs" / f"env_{rank}"
            os.makedirs(env_log_dir, exist_ok=True)
            
            # 传递参数给环境，确保每个环境有自己的日志路径
            current_env_args = copy.copy(all_args)
            current_env_args.log_dir_env = str(env_log_dir)

            env = EnvCore(args=current_env_args, rank=rank)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    # 创建评估环境，保持不变# 
    def get_env_fn(rank):
        def init_env():
            eval_env_log_dir = Path(all_args.log_dir) / "eval_envs" / f"eval_env_{rank}"
            os.makedirs(eval_env_log_dir, exist_ok=True)

            current_eval_env_args = copy.copy(all_args)
            current_eval_env_args.log_dir_env = str(eval_env_log_dir)

            env = EnvCore(args=current_eval_env_args, rank=rank)
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    # 注意: 评估环境通常只需要一个
    return DummyVecEnv([get_env_fn(0)])

def mappo_main(all_args):
    # 主训练函数# 
    # 确保 rollout 线程数与边缘节点数一致
    all_args.n_rollout_threads = all_args.num_edges

    # --- 日志和目录设置，保持不变 ---
    log_dir = Path(all_args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    all_args.log_dir = str(log_dir)

    main_logger = logging.getLogger("MappoMain")
    main_logger.setLevel(logging.INFO)
    main_logger.handlers = []
    main_logger.propagate = False

    log_file_path = log_dir / 'runner_main.log'
    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    main_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    main_logger.addHandler(stream_handler)

    main_logger.info("MappoMain Logger initialized successfully.")
    
    # 打印关键的HAC-SFL超参数，方便追溯
    main_logger.info("="*20 + " HAC-SFL Key Hyperparameters " + "="*20)
    main_logger.info(f"Initial Mu: {all_args.mu_initial}")
    main_logger.info(f"Divergence Target: {all_args.divergence_target}")
    main_logger.info(f"Base Perf Reward Weight: {all_args.w_perf}")
    main_logger.info(f"Entropy Coef: {all_args.entropy_coef}")
    main_logger.info("="*65)


    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert not (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "check recurrent policy!"
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        main_logger.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        main_logger.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 运行目录设置
    run_dir = Path(all_args.log_dir).parent
    
    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "num_agents": all_args.num_agents,
        "logger": main_logger # 传递logger
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()
    
    # 训练结束后关闭环境
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()

if __name__ == '__main__':
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
        print("multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("multiprocessing start method already set.")

    # 解析命令行参数
    parser = args_parser()
    args = parser.parse_args()
    
    args.mu_initial = 0.5            # 提高mu的初始值，加强初始发散抑制
    args.divergence_target = 5.0     # 设定一个明确的发散目标
    
    # 【HAC-SFL 组件一】 激励探索
    args.entropy_coef = 0.05         # 提高熵系数，鼓励智能体探索

    # 运行主函数
    mappo_main(args)