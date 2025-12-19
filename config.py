# file: config.py (Definitive Final Version - with all parameters)

import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser(
        description="HAC-SFL: Hierarchical Autonomic-Control for SFL - Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========================= 1. 实验与系统设置 (Experiment & System) =========================
    group = parser.add_argument_group('Experiment & System')
    group.add_argument("--experiment_name", type=str, default="hac_sfl_evaluation", help="实验名称, 用于日志子目录")
    group.add_argument("--seed", type=int, default=12345, help="全局随机种子")
    group.add_argument("--cuda", action="store_true", default=True, help="如果可用, 则使用CUDA")
    group.add_argument("--log_dir", type=str, default="./logs", help="日志与模型保存的根目录")
    group.add_argument("--model_dir", type=str, default=None, help="加载预训练MAPPO模型的路径")
    group.add_argument('--gpu', type=int, default=0, help='要选择的GPU ID (0, 1, 2, ...)')
    group.add_argument('--cloud_port', type=int, default=9100, help="云服务器监听边缘连接的端口")
    group.add_argument('--server_port_base', type=int, default=7000, help="边缘服务器监听客户端连接的起始端口基址")
    group.add_argument('--test_policy', type=str, default='HAC-SFL',
                       choices=['HAC-SFL', 'MADRL-SFL', 'MHR-SFL', 'ARES', 'RAF-SFL', 'SplitFed', 'FedAvg_Baseline'],
                       help="选择要执行的对比测试策略")

    # ========================= 2. 数据集与模型参数 (Dataset & Model) =========================
    group = parser.add_argument_group('Dataset & Model')
    group.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help="数据集名称")
    group.add_argument('--model_name', type=str, default='resnet18', help="模型架构名称")
    group.add_argument('--input_channels', type=int, default=3, help="模型输入通道数")
    group.add_argument('--output_channels', type=int, default=100, help="模型输出类别数")
    group.add_argument('--data_dir', type=str, default='./data', help="数据集存储目录")

    # ========================= 3. 联邦学习与数据分布 (Federated Learning & Data) =========================
    group = parser.add_argument_group('Federated Learning & Data')
    group.add_argument('--iid', type=int, default=0, help="数据分布方式, 1=IID, 0=Non-IID")
    group.add_argument('--alpha', type=float, default=0.5, help="[Non-IID] 狄利克雷分布的alpha参数")
    group.add_argument('--num_clients', type=int, default=30, help="系统中总的客户端(UAV)数量")
    group.add_argument('--num_edges', type=int, default=4, help="系统中的边缘服务器数量")

    # ========================= 4. SFL 核心训练参数 (SFL Core Training) =========================
    group = parser.add_argument_group('SFL/HFL Core Training')
    group.add_argument('--epochs', type=int, default=200, help="总的训练轮次 (Global SFL Rounds)")
    group.add_argument('--sfl_lr', type=float, default=0.01, help="SFL客户端/边缘端本地训练的学习率 (SGD)")
    group.add_argument('--batch_size', type=int, default=128, help="客户端本地训练时使用的批次大小")
    group.add_argument('--momentum', type=float, default=0.9, help="客户端SGD优化器的动量参数")

    # ========================= 5. 【HAC-SFL】云端宏观策略参数 =========================
    group = parser.add_argument_group('HAC-SFL Cloud Policy')
    group.add_argument('--mu_initial', type=float, default=0.1, help="[HAC-SFL] 分层回拉(mu)的初始值")
    group.add_argument('--mu_max', type=float, default=1.5, help="[HAC-SFL] mu的最大值")
    group.add_argument('--divergence_target', type=float, default=4.0, help="[HAC-SFL] 模型散度的目标阈值")
    group.add_argument('--w_similarity', type=float, default=1.0, help="[HAC-SFL] HWA中模型相似度的权重")
    group.add_argument('--w_data_size', type=float, default=0.2, help="[HAC-SFL] HWA中数据量的权重")
    group.add_argument('--w_perf', type=float, default=5.0, help="[HAC-SFL] 奖励函数中性能(accuracy)的基准权重")
    group.add_argument('--alpha_anchor', type=float, default=0.5, help="[HAC-SFL] 锚点模型更新的平滑系数")

    # ========================= 6. MAPPO 环境与智能体 (MAPPO Env & Agent) =========================
    group = parser.add_argument_group('MAPPO Env & Agent')
    group.add_argument("--env_name", type=str, default="SFL_UAV_Env", help="环境名称")
    group.add_argument("--algorithm_name", type=str, default="rmappo", help="选择使用的多智能体PPO算法变体")
    group.add_argument("--num_agents", type=int, default=5, help="每个边缘环境控制的智能体(UAV)数量")
    group.add_argument("--episode_length", type=int, default=100, help="每个 Major Round (Episode) 包含的SFL全局轮次数")
    group.add_argument("--share_policy", action="store_true", default=True, help="智能体是否共享同一个策略网络")
    group.add_argument("--use_centralized_V", action="store_false", default=True, help="是否使用中心化Critic进行训练")
    group.add_argument("--num_env_steps", type=int, default=1000000, help="[Training Only] 总环境步数")
    group.add_argument("--n_rollout_threads", type=int, default=4, help="[Training Only] 并行环境实例数 (应与num_edges匹配)")
    group.add_argument("--n_training_threads", type=int, default=1, help="[Training Only] 用于PyTorch的训练线程数 (torch.set_num_threads)")
    # MHR-SFL 专属：分配分箱个数（每个维度的离散等级数）
    group.add_argument('--alloc_levels', type=int, default=3, help='[MHR-SFL] 算力/带宽分配的离散分箱数量')
    
    # ========================= 7. MAPPO 神经网络 (MAPPO Network) =========================
    group = parser.add_argument_group('MAPPO Network')
    group.add_argument("--hidden_size", type=int, default=128, help="Actor/Critic网络的隐藏层维度")
    group.add_argument("--layer_N", type=int, default=2, help="Actor/Critic网络的层数")
    group.add_argument("--use_ReLU", action="store_false", default=True, help="是否使用ReLU激活函数")
    
    # ========================= 8. 【测试模式】特定参数 (Evaluation Mode Specific) =========================
    group = parser.add_argument_group('Evaluation Mode Specific')
    group.add_argument("--use_render", action="store_true", default=False, help="激活测试模式 (会调用runner.render())")
    group.add_argument('--static_split_layer', type=int, default=4, help="[SplitFed, RAF-SFL] 固定的模型分割点")
    group.add_argument('--static_local_iters', type=int, default=48, help="[SplitFed, ARES] 固定的本地迭代次数")

    # ========================= 9. 日志与评估兼容性参数 (Logging & Eval Compatibility) =========================
    group = parser.add_argument_group('Logging & Eval Compatibility')
    group.add_argument("--save_interval", type=int, default=1000, help="[兼容性] 模型保存间隔")
    group.add_argument("--log_interval", type=int, default=5, help="[兼容性] 日志记录间隔")
    group.add_argument("--use_eval", action="store_true", default=False, help="[兼容性] 是否启用训练中评估")
    group.add_argument("--eval_interval", type=int, default=25, help="[兼容性] 训练中评估的间隔")
    group.add_argument("--n_eval_rollout_threads", type=int, default=1, help="[兼容性] 用于评估的并行环境数量")
    group.add_argument("--n_render_rollout_threads", type=int, default=1, help="[兼容性] 用于渲染的环境数量")

    # ========================= 10. MAPPO 完整兼容性参数 (Full MAPPO Compatibility) =========================
    # --- 以下参数在测试中不活跃, 但为确保 MAPPO Policy/Trainer/Buffer 对象能成功实例化而保留 ---
    group = parser.add_argument_group('MAPPO Compatibility')
    group.add_argument("--use_linear_lr_decay", action="store_true", default=False)
    group.add_argument("--use_recurrent_policy", action="store_true", default=True)
    group.add_argument("--use_naive_recurrent_policy", action='store_true', default=False)
    group.add_argument("--recurrent_N", type=int, default=1)
    group.add_argument("--use_obs_instead_of_state", action="store_true", default=False)
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--critic_lr", type=float, default=3e-4)
    group.add_argument("--opti_eps", type=float, default=1e-5)
    group.add_argument("--weight_decay", type=float, default=0)
    group.add_argument("--ppo_epoch", type=int, default=5)
    group.add_argument("--num_mini_batch", type=int, default=1)
    group.add_argument("--entropy_coef", type=float, default=0.05)
    group.add_argument("--clip_param", type=float, default=0.2)
    group.add_argument("--use_valuenorm", action="store_false", default=True)
    group.add_argument("--use_feature_normalization", action="store_true", default=True)
    group.add_argument("--gamma", type=float, default=0.99)
    group.add_argument("--gae_lambda", type=float, default=0.95)
    group.add_argument("--use_max_grad_norm", action="store_false", default=True)
    group.add_argument("--max_grad_norm", type=float, default=2.0)
    group.add_argument("--use_huber_loss", action="store_false", default=True)
    group.add_argument("--huber_delta", type=float, default=10.0)
    group.add_argument("--use_proper_time_limits", action="store_true", default=False)
    group.add_argument("--use_value_active_masks", action="store_false", default=True)
    group.add_argument("--use_policy_active_masks", action="store_false", default=True)
    group.add_argument("--gain", type=float, default=0.01)
    group.add_argument("--use_orthogonal", action="store_false", default=True)
    group.add_argument("--data_chunk_length", type=int, default=10)
    group.add_argument("--use_stacked_frames", action="store_true", default=False)
    group.add_argument("--stacked_frames", type=int, default=1)
    group.add_argument("--value_loss_coef", type=float, default=1.0)
    group.add_argument("--use_clipped_value_loss", action="store_false", default=True)
    group.add_argument("--use_popart", action="store_true", default=False)
    group.add_argument("--use_gae", action="store_false", default=True) 

    # ========================= 11. 客户端效用参数 (Client Utility Parameters) =========================
    group = parser.add_argument_group('Client Utility Parameters')
    group.add_argument('--w_data', type=float, default=0.4, help="[Utility] 数据效用(U_data)的权重")
    group.add_argument('--w_sys', type=float, default=0.4, help="[Utility] 系统效用(U_sys)的权重")
    group.add_argument('--w_aou', type=float, default=0.2, help="[Utility] 信息年龄效用(U_AoU)的权重")

    # ========================= 12. MHR-SFL 特定参数 (MHR-SFL Specific) =========================
    group = parser.add_argument_group('MHR-SFL Specific')
    group.add_argument('--proto_lambda', type=float, default=0.01, help="[MHR-SFL] 类原型正则化损失的系数 (lambda)")

    return parser