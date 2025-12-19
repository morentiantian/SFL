# edge.py

from collections import OrderedDict, defaultdict
import random
import threading
import time
import socket
import copy
import torch
from torch import autocast, nn, GradScaler
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from utils.Communicator import Communicator
import numpy as np
import logging
from queue import Queue, Empty

# 导入支持MAPPO决策所需的新模块
import gymnasium as gym
from gymnasium import spaces
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
from models.Simple_split import get_model_structural_features_for_all_splits, get_unified_num_split_options

from models.Simple_split import get_unified_model_constructor, get_unified_split_model_function

def _t2n(x):
    """将torch张量转换为numpy数组 (Convert torch tensor to a numpy array)."""
    return x.detach().cpu().numpy()

# GraphBridge是梯度回传的关键
class GraphBridge(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, input_tensor):
        return input_tensor

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        return grad_output

class EdgeServer(Communicator):
    def __init__(self, args, logger_ref: logging.Logger, val_loader: Optional[DataLoader] = None, policy=None):
        super(EdgeServer, self).__init__()
        self.args = args
        self.edge_id = args.edge_id
        self.logger = logger_ref
        self.device = args.device
        self.model_name = args.model_name
        
        # --- 策略与测试模式 ---
        # --- Policy and Test Mode ---
        self.test_policy = getattr(args, 'test_policy', 'HAC-SFL') # 从args获取当前测试策略 (Get current test policy from args)

        # --- 客户端相关共享资源的专用锁, 防止竞态条件 ---
        # --- Dedicated lock for client-related shared resources to prevent race conditions ---
        self.client_data_lock = threading.Lock()

        # --- 网络连接相关 (由 client_data_lock 保护) ---
        self.client_socket_dict: Dict[int, socket.socket] = {}
        self.client_raw_stats: Dict[int, Dict[str, Any]] = {}
        self.selected_clients_in_order: List[Optional[int]] = []

        # --- SFL 状态 ---
        self.global_model_from_cloud: Optional[nn.Module] = None
        self.macro_policy_package_from_cloud: Dict[str, Any] = {'mu': 0.0, 'is_converged': False}
        self.anchor_model_state_from_cloud: Optional[Dict[str, Any]] = None
        self.aggregated_edge_model_state: Optional[Dict[str, Any]] = None
        self.current_global_round: int = 0
        self.total_edge_interactions_for_global_round: int = 0
        
        # 新增一个成员变量来持久化本地模型状态，解决“失忆”问题
        # Add a new member variable to persist local model state and solve the "amnesia" problem
        self.persistent_local_model_state: Optional[Dict[str, Any]] = None
        self.aggregated_edge_model_delta: Optional[Dict[str, Any]] = None
        
        # [MHR-SFL] 原型回放机制相关
        self.global_prototypes: Optional[Dict[int, torch.Tensor]] = None
        self.aggregated_local_prototypes: Optional[Dict[int, torch.Tensor]] = None
        
        # --- MAPPO & 决策相关 (由 client_data_lock 保护) ---
        self.current_mappo_decisions: Dict[int, Dict[str, Any]] = {}
        # 存储上一轮为每个客户端下发的算力/带宽分配（用于观测替换）
        self.prev_allocations: Dict[int, Dict[str, float]] = {}
        
        # --- 线程安全的结果缓冲区 ---
        self.sfl_interaction_results_buffer: List[Dict[str, Any]] = []
        self.client_final_model_buffer: Dict[int, OrderedDict] = {}
        self.server_final_model_buffer: Dict[int, OrderedDict] = {}
        self.current_interaction_components: Dict[int, Dict[str, Any]] = {} # 存储本轮交互的分割组件 (Store split components of the current interaction)
        self.sfl_results_lock = threading.Lock()

        self.val_loader = val_loader
        self.baseline_validation_loss = float('inf')
        self.server_socket: Optional[socket.socket] = None
        
        # ======================= [核心修改 1: 集成EnvCore功能] =======================
        # 这部分代码将 EnvCore 的功能直接集成到 EdgeServer 中

        # 为奖励函数初始化状态变量
        self.best_accuracy_so_far = 0.0
        self.last_round_accuracy = 0.0
        
        # 定义奖励权重
        self.reward_weights = {
            'w_perf': getattr(args, 'w_perf', 5.0),    # 大幅提高性能权重
            'w_time': getattr(args, 'w_time', 0.01),   # 大幅降低时间惩罚权重
            'w_cost': getattr(args, 'w_cost', 0.05),   # 大幅降低动作成本惩罚权重
        }

        # 获取模型结构特征，用于MAPPO的观测空间
        example_input_tensor = self._get_example_tensor(args)
        model_features_info = get_model_structural_features_for_all_splits(args.model_name, example_input_tensor, args.output_channels, self.logger)
        self.model_structural_features_vector = model_features_info["features_vector"]
        self.local_iteration_options = [4, 8, 16, 24, 32, 64]

        # 初始化内置的MAPPO决策智能体
        self.mappo_policy = None
        self.rnn_states = None

        if self.test_policy in ['HAC-SFL', 'MADRL-SFL', 'MHR-SFL']:
            self.logger.info(f"Policy is {self.test_policy}, initializing built-in MAPPO agent...")
            
            # 定义与训练时一致的观测和动作空间
            obs_dim = 8 + len(self.model_structural_features_vector) + 4 + 1 
            act_dim_split = get_unified_num_split_options(args.model_name, self.logger)
            act_dim_iters = len(self.local_iteration_options)

            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.args.num_agents * obs_dim,), dtype=np.float32)
            # 如果是 MHR-SFL, 在动作空间中为算力与带宽分配加入两个离散维度（每档3个层次）
            if self.test_policy == 'MHR-SFL':
                alloc_levels = 3
                act_space = spaces.MultiDiscrete([act_dim_split, act_dim_iters, alloc_levels, alloc_levels])
            else:
                act_space = spaces.MultiDiscrete([act_dim_split, act_dim_iters])

            # 初始化策略网络
            self.mappo_policy = Policy(self.args, obs_space, share_obs_space, act_space, device=self.device)
            
            # 加载预训练的模型权重
            if self.args.model_dir:
                try:
                    policy_actor_state_dict = torch.load(str(self.args.model_dir) + '/actor.pt', map_location=self.device, weights_only=True)
                    self.mappo_policy.actor.load_state_dict(policy_actor_state_dict)
                    self.mappo_policy.actor.eval()
                    self.logger.info("Built-in MAPPO agent loaded pre-trained model successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to load MAPPO model: {e}", exc_info=True)
                    raise
            else:
                self.logger.warning("No --model_dir provided, MAPPO agent will use randomly initialized weights!")

            # 初始化RNN状态，即智能体的“记忆”
            self.rnn_states = np.zeros((1, self.args.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        # ==============================================================================
        
        self.logger.info(f"Instance for Edge Server {self.edge_id} initialized in '{self.test_policy}' test mode.")

    def _get_example_tensor(self, args):
        if args.dataset in ['cifar10', 'cifar100']:
            return torch.randn(1, args.input_channels, 32, 32)
        else:
            raise ValueError(f"Unsupported dataset for example tensor: {args.dataset}")
            
    def start_listening(self):
        """Creates, binds, and listens on the server socket for the current round."""
        if self.server_socket is not None:
            self.logger.warning("Server socket is already open. Closing it before reopening.")
            self.stop_listening()
        
        try:
            server_ip = '127.0.0.1'
            server_port = getattr(self.args, 'server_port_base', 7000) + self.edge_id
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((server_ip, server_port))
            self.server_socket.listen(self.args.num_clients) # 使用总客户端数作为 backlog
            self.logger.info(f"Edge server {self.edge_id} is now listening on {server_ip}:{server_port} for the {self.current_global_round} round.")
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}", exc_info=True)
            self.server_socket = None
            raise

    def stop_listening(self):
        """Closes the main server listening socket at the end of a round."""
        if self.server_socket:
            self.logger.info(f"Edge server {self.edge_id} is closing its main listening socket.")
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.debug(f"Exception while closing server socket: {e}")
            finally:
                self.server_socket = None

    # ======================================================================================
    # SFL 核心执行逻辑 (SFL Core Execution Logic)
    # ======================================================================================

    def execute_all_sfl_interactions_for_global_round(self) -> Dict[str, Any]:
        self.aggregated_edge_model_state = None
        self.aggregated_edge_model_delta = None

        f_g_k = self.total_edge_interactions_for_global_round
        if self.test_policy == 'FedAvg_Baseline':
            f_g_k = 1

        if not self.global_model_from_cloud:
            self.logger.error("Global model not received from cloud.")
            return {'status': 'error', 'message': 'Global model not received.'}
        
        self.logger.info(f"Edge {self.edge_id}: Starting SFL phase for GR {self.current_global_round + 1}, budget f_g_k={f_g_k}")
        start_time = time.time()
        
        decisions, interaction_mode = self._determine_interaction_plan()
        self.current_mappo_decisions = decisions

        if not decisions:
            self.logger.warning("Failed to determine interaction plan. Skipping.")
            return {'status': 'success_no_train', 'total_time_s': 0}

        initial_model_state_for_round = self.global_model_from_cloud.state_dict()
        current_base_model_state = initial_model_state_for_round
        all_results_this_round = []
        
        participants_for_this_round = list(decisions.keys())

        for i in range(f_g_k):
            if interaction_mode == 'fedavg':
                interaction_results = self.execute_one_fedavg_interaction(current_base_model_state, decisions)
            else: # SFL 模式
                interaction_results = self.execute_one_sfl_interaction(current_base_model_state, decisions, participants_for_this_round)
            
            all_results_this_round.extend(interaction_results)
            
            aggregated_result = self.aggregate_all_uav_models_locally(interaction_mode)
            
            if aggregated_result:
                if interaction_mode == 'fedavg':
                    self.aggregated_edge_model_delta = aggregated_result
                else: # SFL 模式
                    current_base_model_state = aggregated_result

        if interaction_mode == 'fedavg':
            if self.aggregated_edge_model_delta:
                self.logger.info("Applying aggregated delta to the initial model to form the final updated model for evaluation.")
                final_updated_state = OrderedDict()
                for key in initial_model_state_for_round:
                    if key in self.aggregated_edge_model_delta:
                        final_updated_state[key] = initial_model_state_for_round[key].to(self.aggregated_edge_model_delta[key].device) + self.aggregated_edge_model_delta[key]
                    else:
                        final_updated_state[key] = initial_model_state_for_round[key]
                self.aggregated_edge_model_state = final_updated_state
            else:
                self.logger.warning("Aggregation resulted in empty delta. Model was not updated.")
                self.aggregated_edge_model_state = initial_model_state_for_round
        else: # SFL 模式
            self.aggregated_edge_model_state = current_base_model_state

        self.logger.info(f"Edge interaction phase completed for GR {self.current_global_round + 1}.")
        
        # 计算奖励
        # Calculate reward
        reward = self.calculate_reward(
            {'status': 'success', 'total_time_s': time.time() - start_time},
            list(self.current_mappo_decisions.values())
        )
        self.logger.info(f"MAPPO reward for this round: {reward:.4f}")

        return {'status': 'success', 'total_time_s': time.time() - start_time, 'all_interaction_results': all_results_this_round}
    
    def calculate_reward(self, exec_results: Dict, actions: List[Dict]) -> float:
        if not exec_results or exec_results.get('status') != 'success':
            self.logger.warning("[Reward] SFL execution failed, returning heavy penalty.")
            return -10.0

        final_loss, final_accuracy = self.evaluate_on_local_validation_set()
        base_loss = self.baseline_validation_loss
        sfl_round_time = exec_results.get('total_time_s', 120.0)

        if np.isinf(final_loss) or np.isnan(final_loss) or final_loss <= 0:
            self.logger.warning(f"[Reward] Encountered invalid loss: {final_loss}, returning heavy penalty.")
            return -10.0
            
        # 重新设计奖励计算逻辑，放大对准确率的激励
        # Redesign reward calculation logic to amplify incentives for accuracy
        accuracy_gain = final_accuracy - self.last_round_accuracy
        velocity_reward = accuracy_gain * 100 * self.reward_weights['w_perf']

        plateau_bonus = 0.0
        if final_accuracy > self.best_accuracy_so_far:
            plateau_bonus = 100.0  # 突破记录给予非常高的奖励 (Give a very high reward for breaking the record)
            self.best_accuracy_so_far = final_accuracy
        
        loss_reduction = base_loss - final_loss
        loss_reward = max(0, loss_reduction * 20)

        accuracy_penalty = 0.0
        if accuracy_gain < -0.1: # 即使轻微下降也要惩罚 (Penalize even for a slight drop)
            accuracy_penalty = abs(accuracy_gain) * 50 * self.reward_weights['w_perf']
        
        time_penalty = 0.0
        cost_penalty = 0.0
        if final_accuracy > 30.0:
            MAX_REASONABLE_TIME = 120.0
            time_penalty = (sfl_round_time / MAX_REASONABLE_TIME) * self.reward_weights['w_time']
            
            # 成本惩罚的逻辑需要根据actions的实际格式调整，此处暂时简化
            # The logic for cost penalty needs adjustment based on the actual format of actions, simplified here for now
            pass
        
        final_reward = (
            velocity_reward + plateau_bonus + loss_reward -
            accuracy_penalty - time_penalty - cost_penalty
        )

        self.last_round_accuracy = final_accuracy

        self.logger.info(
            f"[Step]: {self.current_global_round + 1} "
            f"[Reward] Final: {final_reward:.4f} (Velo: {velocity_reward:.2f}, Bonus: {plateau_bonus:.2f}, "
            f"LossRwd: {loss_reward:.4f}, AccPenalty: {-accuracy_penalty:.2f}, " 
            f"TimePenalty: {-time_penalty:.4f}, CostPenalty: {-cost_penalty:.4f}) | "
            f"Loss: {final_loss:.4f} | Acc: {final_accuracy:.2f}% (Best: {self.best_accuracy_so_far:.2f}%)"
        )
        return float(final_reward)

    def _select_clients_for_interaction(self) -> List[int]:
        available_clients = list(self.client_socket_dict.keys())
        if not available_clients:
            return []
        num_to_select = self.args.num_agents
        if self.test_policy == 'SplitFed':
            self.logger.info(f"[{self.test_policy} mode] Randomly selecting up to {num_to_select} clients.")
            return random.sample(available_clients, min(len(available_clients), num_to_select))
        else:
            self.logger.info(f"[{self.test_policy} mode] Selecting top {num_to_select} clients based on utility.")
            clients_with_utility = []
            for cid in available_clients:
                stats = self.client_raw_stats.get(cid, {})
                utility = stats.get('utility', 0.0)
                clients_with_utility.append((utility, cid))
            clients_with_utility.sort(key=lambda x: x[0], reverse=True)
            selected_clients = [cid for utility, cid in clients_with_utility[:num_to_select]]
            return selected_clients

    def _get_obs_for_mappo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        individual_obs_list = []
        
        mapped_uav_ids = self.selected_clients_in_order
        cloud_policy_package = self.macro_policy_package_from_cloud
        
        cloud_policy_vector = np.array([
            cloud_policy_package.get('mu', 0.1),
            self.reward_weights.get('w_perf', 1.0),
            self.reward_weights.get('w_time', 0.1),
            self.reward_weights.get('w_cost', 0.2)
        ], dtype=np.float32)

        normalized_training_stage = self.current_global_round / self.args.epochs

        obs_dim = 8 + len(self.model_structural_features_vector) + 4 + 1
        for agent_slot_idx in range(self.args.num_agents):
            client_id = mapped_uav_ids[agent_slot_idx] if agent_slot_idx < len(mapped_uav_ids) else None
            if client_id is not None and client_id in self.client_raw_stats:
                client_features = self.get_client_features_for_mappo(client_id)
                combined_obs = np.concatenate([
                    client_features, 
                    self.model_structural_features_vector, 
                    cloud_policy_vector,
                    np.array([normalized_training_stage])
                ]).astype(np.float32)
                individual_obs_list.append(combined_obs)
            else:
                individual_obs_list.append(np.zeros(obs_dim, dtype=np.float32))
                
        obs = np.array(individual_obs_list)
        share_obs = np.array([obs.flatten()] * self.args.num_agents)
        masks = np.ones((self.args.num_agents, 1), dtype=np.float32)
        
        return obs, share_obs, masks

    # 统一的决策函数，根据self.test_policy选择决策方式
    def _determine_interaction_plan(self) -> Tuple[Dict[int, Any], str]:
        policy = self.test_policy
        decisions = {}
        interaction_mode = 'sfl'
        
        self._request_and_update_all_client_stats()
        self.map_clients_to_agent_slots(self.args.num_agents)
        participating_uav_ids = [cid for cid in self.selected_clients_in_order if cid is not None]

        if not participating_uav_ids:
            return {}, 'none'

        # ======================= [核心修改 2: 集成 MAPPO 决策] =======================
        if policy in ['MADRL-SFL', 'HAC-SFL', 'MHR-SFL']:
            self.logger.info(f"[{policy} mode] Using built-in MAPPO agent for decision making...")
            
            obs_data, share_obs_data, masks_data = self._get_obs_for_mappo()
            
            with torch.no_grad():
                self.mappo_policy.actor.eval()
                actions, _, next_rnn_states = self.mappo_policy.actor(
                    torch.from_numpy(obs_data).to(self.device),
                    torch.from_numpy(self.rnn_states).to(self.device).reshape(-1, self.args.recurrent_N, self.args.hidden_size),
                    torch.from_numpy(masks_data).to(self.device),
                    deterministic=True
                )
                self.rnn_states = _t2n(next_rnn_states.reshape(1, self.args.num_agents, self.args.recurrent_N, self.args.hidden_size))

            actions = _t2n(actions)
            # 定义按 quality_tier 划分的离散等级映射（每档3个层次）
            cpu_levels = {
                'low': [0.5, 1.0, 1.5],
                'medium': [1.5, 2.0, 2.5],
                'high': [2.5, 3.0, 3.5]
            }
            bw_levels = {
                'low': [5, 7, 10],
                'medium': [10, 15, 20],
                'high': [20, 25, 30]
            }

            for i, uav_id in enumerate(participating_uav_ids):
                if i < len(actions):
                    split_layer_idx = int(actions[i][0])
                    local_iters_idx = int(actions[i][1])
                    decision = {
                        'split_layer': split_layer_idx + 1,
                        'local_iterations': self.local_iteration_options[local_iters_idx]
                    }
                    # 如果是 MHR-SFL, 请解析并映射离散的算力/带宽等级
                    if policy == 'MHR-SFL' and actions.shape[1] >= 4:
                        comp_idx = int(actions[i][2])
                        bw_idx = int(actions[i][3])
                        client_stats = self.client_raw_stats.get(uav_id, {})
                        tier = client_stats.get('quality_tier', 'medium')
                        cpu_choice_list = cpu_levels.get(tier, cpu_levels['medium'])
                        bw_choice_list = bw_levels.get(tier, bw_levels['medium']) if isinstance(bw_levels.get(tier), list) else bw_levels['medium']
                        comp_alloc = float(cpu_choice_list[max(0, min(len(cpu_choice_list)-1, comp_idx))])
                        bw_alloc = float(bw_choice_list[max(0, min(len(bw_choice_list)-1, bw_idx))])
                        decision['comp_alloc'] = comp_alloc
                        decision['bw_alloc'] = bw_alloc
                        # 记录为下一轮的观测使用
                        self.prev_allocations[uav_id] = {'comp': comp_alloc, 'bw': bw_alloc}

                    decisions[uav_id] = decision
            return decisions, interaction_mode
        # ==============================================================================
        
        elif policy == 'FedAvg_Baseline':
            interaction_mode = 'fedavg'
            local_iters = getattr(self.args, 'static_local_iters', 10)
            for uav_id in participating_uav_ids:
                decisions[uav_id] = {'local_iterations': local_iters}
        elif policy == 'SplitFed':
            split_point = getattr(self.args, 'static_split_layer', 4)
            local_iters = getattr(self.args, 'static_local_iters', 10)
            for uav_id in participating_uav_ids:
                decisions[uav_id] = {'split_layer': split_point, 'local_iterations': local_iters}
        elif policy == 'ARES':
            local_iters = getattr(self.args, 'static_local_iters', 10)
            for uav_id in participating_uav_ids:
                client_stats = self.client_raw_stats.get(uav_id, {})
                comp_power = client_stats.get('comp_power', 1.0)
                split_point = int(1 + 6 * (comp_power / 3.5))
                decisions[uav_id] = {'split_layer': max(1, min(7, split_point)), 'local_iterations': local_iters}
        elif policy == 'RAF-SFL':
            split_point = getattr(self.args, 'static_split_layer', 4)
            client_times = {uav_id: (1 / self.client_raw_stats.get(uav_id, {}).get('comp_power', 1.0)) + (10 / self.client_raw_stats.get(uav_id, {}).get('bandwidth', 10)) for uav_id in participating_uav_ids}
            if not client_times: return {}, interaction_mode
            slowest_time = max(client_times.values())
            base_iters = 20
            for uav_id, time_est in client_times.items():
                local_iters = int(base_iters * (slowest_time / time_est))
                decisions[uav_id] = {'split_layer': split_point, 'local_iterations': max(10, min(local_iters, 100))}
        else:
            self.logger.error(f"Unknown test_policy: {policy}. No decisions made.")
            return {}, 'none'
            
        if interaction_mode != 'fedavg':
             decisions = self._ensure_decision_fields(decisions)
        return decisions, interaction_mode
    
    def _ensure_decision_fields(self, decisions: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        default_split_layer = getattr(self.args, 'static_split_layer', 4)
        default_local_iters = getattr(self.args, 'static_local_iters', 10)
        for client_id, decision in decisions.items():
            if not isinstance(decision, dict):
                self.logger.warning(f"Invalid decision format for client {client_id}. Applying defaults.")
                decisions[client_id] = {'split_layer': default_split_layer, 'local_iterations': default_local_iters}
                continue
            if 'split_layer' not in decision:
                decision['split_layer'] = default_split_layer
            if 'local_iterations' not in decision:
                decision['local_iterations'] = default_local_iters
        return decisions

    def _execute_interaction_with_deadline(self, target_func, base_model_state, decisions, participants):
        self._clear_buffers()
        threads = []
        results_queue = Queue()

        with self.client_data_lock:
            connected_clients = dict(self.client_socket_dict)
            standby_clients = {cid: sock for cid, sock in connected_clients.items() if cid not in participants}
            
            for uav_id, sock in standby_clients.items():
                try:
                    self.send_msg(sock, {'participate': False, 'current_global_round': self.current_global_round}, 'MSG_TRAINING_DIRECTIVE_TO_CLIENT')
                except Exception as e:
                    self.logger.warning(f"Failed to send standby directive to client {uav_id}: {e}")

            for uav_id in participants:
                if uav_id in connected_clients and uav_id in decisions:
                    client_sock_ref = connected_clients[uav_id]
                    # 在 directive 中注入 comp_alloc / bw_alloc（如果决策里包含）
                    dir_decision = dict(decisions[uav_id])
                    thread_args = (uav_id, client_sock_ref, base_model_state, dir_decision, results_queue)
                    thread = threading.Thread(target=target_func, args=thread_args)
                    threads.append((thread, uav_id))
                    thread.start()

        deadline = time.time() + 580
        
        completed_clients = set()
        started_clients = {uav_id for _, uav_id in threads}
        num_expected = len(started_clients)

        while len(completed_clients) < num_expected and time.time() < deadline:
            try:
                result = results_queue.get(timeout=1.0)
                client_id = result.get('uav_id')
                
                if client_id is None: continue
                
                completed_clients.add(client_id)
                
                if result.get('status') == 'success':
                    with self.sfl_results_lock:
                        if 'model_delta' in result:
                            self.client_final_model_buffer[result['uav_id']] = result['model_delta']
                        elif 'client_model' in result:
                            self.client_final_model_buffer[result['uav_id']] = result['client_model']
                            self.server_final_model_buffer[result['uav_id']] = result['server_model']
                        
                        if 'label_counts' in result:
                            if result['uav_id'] not in self.client_raw_stats:
                                self.client_raw_stats[result['uav_id']] = {}
                            self.client_raw_stats[result['uav_id']]['label_counts'] = result['label_counts']

                        self.sfl_interaction_results_buffer.append(result)
                else:
                    self.logger.warning(f"Client {client_id} reported failure: {result.get('error')}")
                    self.safely_remove_client(client_id)
            except Empty:
                continue

        timed_out_clients = started_clients - completed_clients
        if timed_out_clients:
            self.logger.warning(f"Interaction deadline reached. The following clients timed out and will be removed: {list(timed_out_clients)}")
            for uav_id in timed_out_clients:
                self.safely_remove_client(uav_id)
        
        return copy.deepcopy(self.sfl_interaction_results_buffer)
    
    def accept_persistent_connections(self, num_expected_clients: int, timeout: int = 120):
        self.start_listening()
        
        if self.server_socket is None:
            self.logger.error("Cannot accept clients because the server socket failed to start listening.")
            return 0
        
        self.server_socket.settimeout(2.0) 
        self.logger.info(f"Edge {self.edge_id}: [Persistent Connection Phase] Accepting connections for the entire episode...")
        self.logger.info(f"Waiting for all {num_expected_clients} assigned clients to connect (timeout: {timeout}s)...")
        start_time = time.time()
        handshake_threads = []

        while len(self.client_socket_dict) < num_expected_clients and time.time() - start_time < timeout:
            try:
                sock, addr = self.server_socket.accept()
                thread = threading.Thread(target=self._handle_client_handshake_thread, args=(sock, addr))
                thread.daemon = True
                thread.start()
                handshake_threads.append(thread)
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Error while accepting new connection: {e}", exc_info=True)
        
        self.logger.info(f"Connection window closed. Waiting for {len(handshake_threads)} handshake threads to complete...")
        for t in handshake_threads:
            t.join(timeout=10.0)

        with self.client_data_lock:
            num_connected_clients = len(self.client_socket_dict)

        self.stop_listening()
        
        if num_connected_clients < num_expected_clients:
            self.logger.warning(f"Only {num_connected_clients}/{num_expected_clients} clients connected. Proceeding with available clients.")
        else:
            self.logger.info(f"All {num_connected_clients} clients are now persistently connected.")
        
        return num_connected_clients
    
    def reset_for_new_episode(self):
        self.logger.info(f"Edge {self.edge_id}: Resetting state for new episode.")
        self.stop_listening() 
        
        with self.client_data_lock:
            self.current_mappo_decisions.clear()
            self.selected_clients_in_order.clear()
        with self.sfl_results_lock:
            self._clear_buffers()
            
        self.persistent_local_model_state = None
        self.aggregated_edge_model_state = None
        self.baseline_validation_loss = float('inf')

        if self.rnn_states is not None:
            self.rnn_states = np.zeros_like(self.rnn_states)
            self.logger.info("MAPPO agent RNN states have been reset.")


    def execute_one_sfl_interaction(self, base_model_state: dict, decisions: dict, participants_this_interaction: List[int]) -> List[Dict]:
        return self._execute_interaction_with_deadline(
            self._handle_single_uav_sfl_thread, base_model_state, decisions, participants_this_interaction
        )
        
    def _handle_single_uav_sfl_thread(self, client_id: int, client_sock: socket.socket, base_model_state: dict, decision: dict, results_queue: Queue):
        try:
            total_iterations = decision['local_iterations']

            sfl_params = self._prepare_sfl_thread_resources(decision['split_layer'], base_model_state)
            if not sfl_params:
                raise RuntimeError("Failed to prepare SFL resources.")
            
            with self.sfl_results_lock:
                self.current_interaction_components[client_id] = {'key_maps': sfl_params.get('key_maps', {})}
            
            server_model_part = sfl_params['server_model_part']
            optimizer = sfl_params['optimizer']
            
            directive_to_client = self._create_sfl_directive(sfl_params, total_iterations, decision)
            if not self.send_msg(client_sock, directive_to_client, 'MSG_TRAINING_DIRECTIVE_TO_CLIENT'):
                raise ConnectionError("Failed to send SFL directive.")
            
            server_model_part.train()
            total_loss = 0.0
            scaler = GradScaler(enabled=(self.device.type == 'cuda'))
            total_label_counts = {}
            
            # [MHR-SFL] 本地原型累加器
            local_proto_sums = {}
            local_proto_counts = {}

            for i in range(total_iterations):
                msg_type, smashed_data_msg = self.recv_msg(client_sock, timeout=600.0)
                if not smashed_data_msg or msg_type != 'MSG_SMASHED_DATA_TO_EDGE':
                    raise ConnectionError(f"Did not receive valid smashed data on iteration {i+1}.")

                client_output = smashed_data_msg['outputs'].to(self.device).requires_grad_(True)
                label = smashed_data_msg['label'].to(self.device)
                client_output.retain_grad()
                
                # 统计当前batch的类别分布 (Accumulate label counts for aggregation)
                labels_cpu = label.cpu().numpy()
                unique_labels, counts = np.unique(labels_cpu, return_counts=True)
                for l, c in zip(unique_labels, counts):
                    total_label_counts[int(l)] = total_label_counts.get(int(l), 0) + int(c)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                    server_input = GraphBridge.apply(client_output)
                    
                    # [MHR-SFL] 特征提取与原型正则化
                    features = None
                    output_server = None
                    
                    if self.args.test_policy == 'MHR-SFL':
                        # 尝试分离特征提取器和分类器
                        modules = list(server_model_part.children())
                        if len(modules) > 1:
                            feature_extractor = nn.Sequential(*modules[:-1])
                            classifier = modules[-1]
                            features = feature_extractor(server_input.float())
                            output_server = classifier(features)
                        else:
                            features = server_input.float()
                            output_server = server_model_part(features)
                    else:
                        output_server = server_model_part(server_input.float())

                    # [MHR-SFL] 动态计算类别权重 (Class-Balanced Loss)
                    if self.args.test_policy == 'MHR-SFL':
                        # ... (Existing CB Loss logic) ...
                        # unique_labels, counts already calculated above
                        
                        # 获取总类别数 (从args中获取)
                        num_classes = self.args.output_channels
                        
                        beta = 0.5 # beta \in (0, 1]
                        class_counts = dict(zip(unique_labels, counts))
                        denom = sum(float(class_counts[c]) ** (-beta) for c in unique_labels)
                        weight_tensor = torch.ones(num_classes, device=self.device)
                        
                        for c, n_c in class_counts.items():
                            w_c = (float(n_c) ** (-beta)) / denom
                            weight_tensor[c] = w_c * len(unique_labels) 
                            
                        cls_loss = nn.CrossEntropyLoss(weight=weight_tensor)(output_server, label)
                        
                        # [MHR-SFL] 添加原型正则化损失
                        reg_loss = 0.0
                        if self.global_prototypes:
                            batch_protos = []
                            valid_mask = []
                            for idx, y in enumerate(label):
                                y_item = int(y.item())
                                if y_item in self.global_prototypes:
                                    batch_protos.append(self.global_prototypes[y_item])
                                    valid_mask.append(idx)
                            
                            if batch_protos:
                                batch_protos_tensor = torch.stack(batch_protos).to(self.device)
                                selected_features = features[valid_mask]
                                # 正则化系数 (从args获取，默认为0.01，防止掩盖主损失)
                                lambda_proto = getattr(self.args, 'proto_lambda', 0.01)
                                reg_loss = lambda_proto * nn.MSELoss()(selected_features, batch_protos_tensor)
                        
                        loss = cls_loss + reg_loss
                        
                        # [MHR-SFL] 累积本地原型
                        features_detached = features.detach()
                        for idx, y in enumerate(label):
                            y_item = int(y.item())
                            if y_item not in local_proto_sums:
                                local_proto_sums[y_item] = torch.zeros_like(features_detached[idx])
                                local_proto_counts[y_item] = 0
                            local_proto_sums[y_item] += features_detached[idx]
                            local_proto_counts[y_item] += 1
                            
                    else:
                        loss = nn.CrossEntropyLoss()(output_server, label)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(server_model_part.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                
                client_grad = client_output.grad.clone().detach() if client_output.grad is not None else torch.zeros_like(client_output)
                if not self.send_msg(client_sock, {'gradient': client_grad.cpu()}, 'MSG_GRADIENT_TO_CLIENT'):
                    raise ConnectionError(f"Failed to send gradient on iteration {i+1}.")
                total_loss += loss.item()

            msg_type, final_report = self.recv_msg(client_sock, timeout=600.0)
            if not final_report or msg_type != 'MSG_FINAL_CLIENT_REPORT_TO_EDGE':
                raise ConnectionError("Did not receive final valid report.")
            
            if not self.send_msg(client_sock, {'status': 'acknowledged_standby'}, 'MSG_EDGE_ACK_TO_CLIENT'):
                raise ConnectionError("Failed to send ACK to client.")
            
            # [MHR-SFL] 计算最终的本地原型 (Mean)
            final_local_prototypes = {}
            if self.args.test_policy == 'MHR-SFL':
                for cls_id, sum_vec in local_proto_sums.items():
                    count = local_proto_counts[cls_id]
                    if count > 0:
                        final_local_prototypes[cls_id] = sum_vec / count

            results_queue.put({
                'uav_id': client_id, 'status': 'success',
                'client_model': final_report['model_state_dict'], 'server_model': server_model_part.state_dict(),
                'avg_loss': total_loss / total_iterations if total_iterations > 0 else 0,
                'client_kpi_report': final_report.get('kpi_report', {}),
                'label_counts': total_label_counts,
                'local_prototypes': final_local_prototypes # [MHR-SFL]
            })
        except Exception as e:
            self.logger.warning(f"SFL thread for UAV {client_id} failed: {e}")
            results_queue.put({'uav_id': client_id, 'status': 'failed', 'error': str(e)})

    def execute_one_fedavg_interaction(self, base_model_state: dict, decisions: dict) -> List[Dict]:
        return self._execute_interaction_with_deadline(
            self._handle_single_uav_fedavg_thread, base_model_state, decisions, list(decisions.keys())
        )

    def _handle_single_uav_fedavg_thread(self, client_id: int, client_sock: socket.socket, model_state: dict, decision: dict, results_queue: Queue):
        try:
            msg_to_client = {
                'participate': True, 'training_mode': 'fedavg',
                'model_state_dict': model_state, 'local_iterations': decision['local_iterations'],
                'current_global_round': self.current_global_round,
                'sfl_lr': self.macro_policy_package_from_cloud.get('sfl_lr', self.args.sfl_lr)
            }
            if not self.send_msg(client_sock, msg_to_client, 'MSG_TRAINING_DIRECTIVE_TO_CLIENT'):
                raise ConnectionError("Failed to send FedAvg directive.")
            msg_type, final_report = self.recv_msg(client_sock, timeout=900.0)
            if not final_report or msg_type != 'MSG_FINAL_CLIENT_REPORT_TO_EDGE' or 'model_delta' not in final_report:
                raise ConnectionError("Did not receive valid final FedAvg report with model_delta.")
            results_queue.put({
                'uav_id': client_id, 'status': 'success',
                'model_delta': final_report['model_delta'],
                'client_kpi_report': final_report.get('kpi_report', {})
            })
        except Exception as e:
            self.logger.warning(f"FedAvg thread for UAV {client_id} failed: {e}")
            results_queue.put({'uav_id': client_id, 'status': 'failed', 'error': str(e)})

    # ======================================================================================
    # 辅助函数与管理逻辑 (Helper Functions and Management Logic)
    # ======================================================================================

    def disconnect_all_clients(self):
        with self.client_data_lock:
            if not self.client_socket_dict:
                return
            self.logger.info(f"Edge {self.edge_id}: Disconnecting all ({len(self.client_socket_dict)}) clients to signal end of round.")
            sockets_to_close = list(self.client_socket_dict.values())
            for sock in sockets_to_close:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except OSError:
                    pass
                except Exception as e:
                    self.logger.debug(f"Exception while closing client socket: {e}")
            self.client_socket_dict.clear()
            self.client_raw_stats.clear()

    def map_clients_to_agent_slots(self, num_slots: int):
        with self.client_data_lock:
            clients_with_utility = []
            for cid in self.client_socket_dict.keys():
                utility = self.client_raw_stats.get(cid, {}).get('utility', 0.0)
                clients_with_utility.append((utility, cid))
            clients_with_utility.sort(key=lambda x: x[0], reverse=True)
            sorted_clients = [cid for _, cid in clients_with_utility]
            self.selected_clients_in_order = sorted_clients[:num_slots]
            while len(self.selected_clients_in_order) < num_slots:
                self.selected_clients_in_order.append(None)
            self.logger.info(f"Mapped clients to agent slots: {self.selected_clients_in_order}")

    def _request_and_update_all_client_stats(self):
        self.logger.info("Requesting status updates from all connected clients...")
        threads = []
        lock = threading.Lock()

        def collect_status(client_id, sock):
            try:
                if not self.send_msg(sock, {}, 'MSG_REQUEST_STATUS_UPDATE'):
                    raise ConnectionError("Failed to send status update request.")
                
                msg_type, response = self.recv_msg(sock, timeout=10.0)
                if msg_type == 'MSG_CLIENT_STATUS_UPDATE' and response and 'status_payload' in response:
                    with lock:
                        self.client_raw_stats[client_id] = response['status_payload']
                    self.logger.debug(f"Successfully updated status for client {client_id}.")
                else:
                    raise ConnectionError(f"Invalid or missing status update response. Type: {msg_type}")
            except Exception as e:
                self.logger.warning(f"Failed to get status update from client {client_id}: {e}. It may be disconnected.")
                self.safely_remove_client(client_id)

        with self.client_data_lock:
            for cid, client_sock in self.client_socket_dict.items():
                thread = threading.Thread(target=collect_status, args=(cid, client_sock))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()
        
        self.logger.info("Finished collecting status updates from clients.")

    def get_client_features_for_mappo(self, client_id: int) -> np.ndarray:
        stats = self.client_raw_stats.get(client_id, {})
        norm_data_size = stats.get('data_size', 0) / 5000.0
        # 如果存在上一轮的分配，则用上一轮分配替代原始的 comp/bandwidth 观测
        prev = self.prev_allocations.get(client_id)
        if prev is not None:
            norm_comp_power = prev.get('comp', stats.get('comp_power', 0)) / 3.5
            norm_bandwidth = prev.get('bw', stats.get('bandwidth', 0)) / 30.0
        else:
            norm_comp_power = stats.get('comp_power', 0) / 3.5
            norm_bandwidth = stats.get('bandwidth', 0) / 30.0
        norm_link_rate = stats.get('link_rate', 0) / 500.0
        norm_battery = stats.get('battery', 0)
        norm_distance = stats.get('distance', 0) / 500.0
        norm_last_round = (self.current_global_round - stats.get('last_round_participated', -1)) / self.args.epochs
        norm_utility = stats.get('utility', 0)
        return np.array([
            norm_data_size, norm_comp_power, norm_bandwidth, norm_link_rate,
            norm_battery, norm_distance, norm_last_round, norm_utility
        ], dtype=np.float32)

    def _clear_buffers(self):
        self.sfl_interaction_results_buffer.clear()
        self.client_final_model_buffer.clear()
        self.server_final_model_buffer.clear()
        self.current_interaction_components.clear()

    def _prepare_sfl_thread_resources(self, split_point: int, base_model_state: dict) -> Optional[Dict[str, Any]]:
        try:
            full_model = get_unified_model_constructor(self.model_name, self.args.input_channels, self.args.output_channels)
            full_model.load_state_dict(base_model_state)
            client_part, server_part, client_map, server_map = get_unified_split_model_function(full_model, self.model_name, split_point)
            lr = self.macro_policy_package_from_cloud.get('sfl_lr', self.args.sfl_lr)
            optimizer = optim.SGD(server_part.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            return {
                'client_model_part_state_dict': {k: v.cpu() for k, v in client_part.state_dict().items()},
                'server_model_part': server_part.to(self.device),
                'optimizer': optimizer,
                'split_layer': int(split_point),
                'key_maps': {'client_key_map': client_map, 'server_key_map': server_map}
            }
        except Exception as e:
            self.logger.error(f"Error preparing resources for split_point {split_point}: {e}", exc_info=True)
            return None
    
    def _create_sfl_directive(self, sfl_params: Dict[str, Any], num_iterations: int, decision: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mu = self.macro_policy_package_from_cloud.get('mu', 0.0)
        split_anchor_state = None
        if self.anchor_model_state_from_cloud and mu > 0:
            client_key_map = sfl_params['key_maps']['client_key_map']
            split_anchor_state = OrderedDict((k, self.anchor_model_state_from_cloud[v].cpu()) for k, v in client_key_map.items() if v in self.anchor_model_state_from_cloud)
        directive = {
            'participate': True, 'training_mode': 'sfl',
            'model_split_point': sfl_params['split_layer'],
            'client_model_part_state_dict': sfl_params['client_model_part_state_dict'],
            'local_iterations': num_iterations,
            'anchor_model_for_pullback': split_anchor_state, 'mu': mu,
            'current_global_round': self.current_global_round,
            'sfl_lr': self.args.sfl_lr
        }
        # 如果决策中包含算力/带宽分配，则下发给客户端用于本次交互
        if decision is not None:
            if 'comp_alloc' in decision:
                directive['comp_alloc'] = decision['comp_alloc']
            if 'bw_alloc' in decision:
                directive['bw_alloc'] = decision['bw_alloc']
        return directive
    
    def _calculate_mhr_weights(self, uids: List[int]) -> List[float]:
        # 1. Data Volume
        data_sizes = np.array([self.client_raw_stats.get(uid, {}).get('data_size', 1) for uid in uids])
        
        # 2. Distribution Consistency
        # Gather label counts
        all_label_counts = []
        for uid in uids:
            counts = self.client_raw_stats.get(uid, {}).get('label_counts', {})
            # Convert to vector of size output_channels
            vec = np.zeros(self.args.output_channels)
            for l, c in counts.items():
                if int(l) < self.args.output_channels:
                    vec[int(l)] = c
            all_label_counts.append(vec)
        
        all_label_counts = np.array(all_label_counts)
        group_sum = np.sum(all_label_counts, axis=0)
        group_dist = group_sum / (np.sum(group_sum) + 1e-8)
        
        dist_scores = []
        for vec in all_label_counts:
            p_k = vec / (np.sum(vec) + 1e-8)
            # RMSE
            rmse = np.sqrt(np.mean((p_k - group_dist)**2))
            dist_scores.append(np.exp(-5.0 * rmse)) # lambda=5.0
            
        # Combine
        final_weights = []
        for i in range(len(uids)):
            w = data_sizes[i] * dist_scores[i]
            final_weights.append(w)
            
        return final_weights

    def aggregate_all_uav_models_locally(self, interaction_mode: str) -> Optional[Dict[str, Any]]:
        with self.sfl_results_lock:
            if not self.client_final_model_buffer:
                self.logger.warning("[Aggregate] Client final model buffer is empty. Skipping local aggregation.")
                return None

            if interaction_mode == 'fedavg':
                self.logger.info("[Aggregate] Performing FedAvg aggregation for model deltas.")
                client_deltas = list(self.client_final_model_buffer.values())
                weights = [self.client_raw_stats.get(uid, {}).get('data_size', 1) 
                        for uid in self.client_final_model_buffer.keys()]
                aggregated_delta = self.average_state_dicts(client_deltas, weights)
                if not aggregated_delta:
                    self.logger.error("[Aggregate] FedAvg delta aggregation failed.")
                    return None
                self.logger.info(f"Successfully aggregated {len(client_deltas)} model deltas using FedAvg.")
                return aggregated_delta

            elif interaction_mode == 'sfl':
                self.logger.info("[Aggregate] Performing SFL aggregation for split models.")
                grouped_clients_by_sp: Dict[int, List[int]] = {}
                for uid, decision in self.current_mappo_decisions.items():
                    if uid in self.client_final_model_buffer:
                        split_point = decision.get('split_layer')
                        if split_point is None:
                            self.logger.warning(f"Client {uid} participated but has no split_layer in decisions. Skipping.")
                            continue
                        if split_point not in grouped_clients_by_sp:
                            grouped_clients_by_sp[split_point] = []
                        grouped_clients_by_sp[split_point].append(uid)
                
                if not grouped_clients_by_sp:
                    self.logger.warning("[Aggregate] No valid client groups to aggregate after filtering. Aborting.")
                    return None
                
                self.logger.info(f"[Aggregate] Clients grouped by split points: { {sp: len(uids) for sp, uids in grouped_clients_by_sp.items()} }")

                recombined_models_from_groups = []
                final_aggregation_weights = []

                for sp, uids_in_group in grouped_clients_by_sp.items():
                    if not uids_in_group: continue
                    client_states = [self.client_final_model_buffer[uid] for uid in uids_in_group]
                    server_states = [self.server_final_model_buffer[uid] for uid in uids_in_group]
                    
                    if self.args.test_policy == 'MHR-SFL':
                        intra_group_weights = self._calculate_mhr_weights(uids_in_group)
                    else:
                        intra_group_weights = [self.client_raw_stats.get(uid, {}).get('data_size', 1) for uid in uids_in_group]
                        
                    avg_client_state = self.average_state_dicts(client_states, weights=intra_group_weights)
                    avg_server_state = self.average_state_dicts(server_states, weights=intra_group_weights)

                    if not avg_client_state or not avg_server_state:
                        self.logger.warning(f"Averaging client or server states failed for group with SP={sp}. Skipping group.")
                        continue
                    
                    representative_uid = uids_in_group[0]
                    key_maps = self.current_interaction_components[representative_uid].get('key_maps')
                    if not key_maps:
                        self.logger.warning(f"Key maps not found for SP={sp} using client {representative_uid}. Skipping group.")
                        continue
                        
                    recombined_state_for_group = self._recombine_model_parts(
                        avg_client_state, avg_server_state, key_maps['client_key_map'], key_maps['server_key_map']
                    )
                    
                    if not recombined_state_for_group:
                        self.logger.warning(f"Recombining model failed for group with SP={sp}. Skipping group.")
                        continue
                    
                    recombined_models_from_groups.append(recombined_state_for_group)
                    
                    # [MHR-SFL Correction] 
                    # Intra-group aggregation uses MHR weights (Data * Consistency).
                    # Inter-group aggregation uses pure Data Volume sum.
                    if self.args.test_policy == 'MHR-SFL':
                        group_total_weight = sum([self.client_raw_stats.get(uid, {}).get('data_size', 1) for uid in uids_in_group])
                    else:
                        group_total_weight = sum(intra_group_weights)
                        
                    final_aggregation_weights.append(group_total_weight)
                    self.logger.info(f"Successfully recombined model for group with SP={sp}, total weight={group_total_weight}.")

                if not recombined_models_from_groups:
                    self.logger.error("[Aggregate] No valid models were recombined across all groups. Aggregation failed.")
                    return None

                final_aggregated_state = self.average_state_dicts(recombined_models_from_groups, weights=final_aggregation_weights)
                
                if not final_aggregated_state:
                    self.logger.error("[Aggregate] Final aggregation of group models failed.")
                    return None

                for k, v in final_aggregated_state.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        self.logger.error(f"Invalid tensor values (NaN or Inf) found in FINAL aggregated state at key: '{k}'. Aggregation failed.")
                        return None
                
                # [MHR-SFL] 聚合本地原型 (Aggregate Local Prototypes)
                if self.args.test_policy == 'MHR-SFL':
                    agg_proto_sums = {}
                    agg_proto_counts = {}
                    
                    for res in self.sfl_interaction_results_buffer:
                        if res['status'] != 'success': continue
                        
                        local_protos = res.get('local_prototypes', {})
                        label_counts = res.get('label_counts', {})
                        
                        for cls_id, mean_vec in local_protos.items():
                            # label_counts keys might be int, ensure consistency
                            count = label_counts.get(int(cls_id), 0)
                            if count == 0: continue
                            
                            if cls_id not in agg_proto_sums:
                                agg_proto_sums[cls_id] = torch.zeros_like(mean_vec)
                                agg_proto_counts[cls_id] = 0
                            
                            agg_proto_sums[cls_id] += mean_vec * count
                            agg_proto_counts[cls_id] += count
                            
                    self.aggregated_local_prototypes = {}
                    for cls_id, sum_vec in agg_proto_sums.items():
                        if agg_proto_counts[cls_id] > 0:
                            self.aggregated_local_prototypes[cls_id] = sum_vec / agg_proto_counts[cls_id]
                    
                    self.logger.info(f"[MHR-SFL] Aggregated local prototypes for {len(self.aggregated_local_prototypes)} classes.")

                self.logger.info(f"Successfully performed final aggregation across {len(recombined_models_from_groups)} groups.")
                return final_aggregated_state
                
            else:
                self.logger.error(f"Unknown interaction_mode: '{interaction_mode}'. Cannot aggregate.")
                return None

    def connect_to_cloud(self, cloud_host: str, cloud_port: int, retry: int = 5, delay: int = 3) -> bool:
        for attempt in range(retry):
            try:
                self.cloud_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.cloud_socket.connect((cloud_host, cloud_port))
                self.logger.info(f"Edge {self.edge_id}: Successfully connected to Cloud at {cloud_host}:{cloud_port}")
                if self.send_msg(self.cloud_socket, {'edge_id': self.edge_id}, 'MSG_EDGE_HELLO'):
                    return True
                self.cloud_socket.close()
            except Exception as e:
                self.logger.warning(f"Edge {self.edge_id}: Attempt {attempt+1}/{retry} to connect to cloud failed: {e}")
                time.sleep(delay)
        return False

    def receive_global_config_from_cloud(self) -> Tuple[bool, int]:
        try:
            for _ in range(2):
                msg_type, cloud_decision = self.recv_msg(self.cloud_socket, timeout=700.0)

                if cloud_decision and msg_type == "MSG_CLOUD_DECISION_TO_EDGE":
                    self.current_global_round = cloud_decision.get('current_global_round', self.current_global_round)
                    self.total_edge_interactions_for_global_round = cloud_decision.get('edge_rounds', 0)
                    self.anchor_model_state_from_cloud = {k: v.cpu() for k, v in cloud_decision.get('anchor_model_state', {}).items()} if cloud_decision.get('anchor_model_state') else None
                    self.macro_policy_package_from_cloud = cloud_decision.get('macro_policy_package', self.macro_policy_package_from_cloud)
                    
                    # [MHR-SFL] 接收全局原型
                    if 'global_prototypes' in cloud_decision and cloud_decision['global_prototypes']:
                        self.global_prototypes = {k: v.to(self.device) for k, v in cloud_decision['global_prototypes'].items()}
                    else:
                        self.global_prototypes = None

                    if 'model_state_dict' not in cloud_decision:
                        self.logger.error("Invalid config from cloud: missing model state.")
                        return False, 0
                    
                    if self.global_model_from_cloud is None:
                        self.global_model_from_cloud = get_unified_model_constructor(self.model_name, self.args.input_channels, self.args.output_channels)
                    
                    self.global_model_from_cloud.load_state_dict(cloud_decision['model_state_dict'])
                    self.global_model_from_cloud.to(self.device)
                    return True, self.total_edge_interactions_for_global_round
                
                elif msg_type == 'MSG_CLOUD_ROUND_COMPLETE':
                    self.logger.warning("Received a stray 'ROUND_COMPLETE' signal. Ignoring it and trying to receive again.")
                    continue
                
                else:
                    self.logger.error(f"Received an unexpected message type: {msg_type}")
                    break

            self.logger.error(f"Edge {self.edge_id}: Failed to receive valid config from cloud after handling potential stray messages.")
            return False, 0

        except socket.timeout:
            self.logger.error("Timeout waiting for configuration from the cloud. The cloud server may be stuck or disconnected.")
            return False, 0
        except Exception as e:
            self.logger.error(f"An error occurred while receiving config from cloud: {e}", exc_info=True)
            return False, 0

    def finalize_and_report_to_cloud(self) -> bool:
        if not self.cloud_socket: 
            return False
        
        final_kpi_report = self._aggregate_interaction_kpis()
        
        if self.test_policy == 'FedAvg_Baseline':
            payload_content = self.aggregated_edge_model_delta
            payload_key = 'model_delta'
        else:
            payload_content = self.aggregated_edge_model_state
            payload_key = 'model_state'
            if payload_content:
                final_kpi_report['model_divergence'] = self._calculate_model_divergence()

        if payload_content is None:
            self.logger.warning(f"No aggregated content ({payload_key}) to report. Sending empty update.")
            payload_content = {}
            
        # [MHR-SFL] 上报本地原型
        local_protos_to_send = None
        if self.test_policy == 'MHR-SFL' and self.aggregated_local_prototypes:
            local_protos_to_send = {k: v.cpu() for k, v in self.aggregated_local_prototypes.items()}

        with self.client_data_lock:
            num_active_clients = len(self.client_socket_dict)
            
        msg_to_cloud = {
            'edge_id': self.edge_id,
            'model_payload': {
                payload_key: payload_content, 
                'kpi_report': final_kpi_report,
                'local_prototypes': local_protos_to_send # [MHR-SFL]
            },
            'next_edge_state': {'num_active_clients': num_active_clients, 'kpi_report': final_kpi_report}
        }
        
        if self.send_msg(self.cloud_socket, msg_to_cloud, 'MSG_LOCAL_MODEL_TO_CLOUD'):
            self.logger.info(f"Edge {self.edge_id}: Final report sent to cloud.")
            return True
        else:
            self.logger.error(f"Edge {self.edge_id}: Failed to send report to cloud.")
            return False
            
    def wait_for_round_completion_signal(self):
        self.logger.info(f"Edge {self.edge_id}: Waiting for 'ROUND_COMPLETE' signal from cloud...")
        try:
            msg_type, signal_msg = self.recv_msg(self.cloud_socket, timeout=900.0)
            if msg_type == 'MSG_CLOUD_ROUND_COMPLETE' and signal_msg.get('status') == 'ROUND_COMPLETE':
                return True
            else:
                self.logger.warning(f"Edge {self.edge_id}: Expected 'ROUND_COMPLETE' signal, but received type '{msg_type}'.")
                return False
        except socket.timeout:
            self.logger.error(f"Edge {self.edge_id}: Timed out while waiting for 'ROUND_COMPLETE' signal. The system may be deadlocked.")
            return False
        except Exception as e:
            self.logger.error(f"Edge {self.edge_id}: Error while waiting for 'ROUND_COMPLETE' signal: {e}", exc_info=True)
            return False

    def _aggregate_interaction_kpis(self) -> Dict[str, Any]:
        aggregated_kpis = defaultdict(float)
        successful_interactions = [res for res in self.sfl_interaction_results_buffer if res['status'] == 'success']
        num_successful = len(successful_interactions)
        if num_successful == 0: return dict(aggregated_kpis)
        
        kpi_keys_to_sum = ['total_data_size', 'computation_energy_J', 'communication_energy_J', 'total_energy_consumption_J']
        kpi_keys_to_avg = ['utility', 'computation_time_s', 'communication_time_s', 'total_interaction_time_s', 'idle_time_s']
        
        all_reports = [res.get('client_kpi_report', {}) for res in successful_interactions]
        for key in kpi_keys_to_sum:
            aggregated_kpis[f"total_{key}"] = sum(r.get(key, 0) for r in all_reports)
        for key in kpi_keys_to_avg:
            valid_reports = [r.get(key, 0) for r in all_reports if r.get(key, 0) is not None]
            if valid_reports:
                aggregated_kpis[f"avg_{key}"] = np.mean(valid_reports)
        
        self.logger.info(f"Aggregated KPIs from {num_successful} clients for this round.")
        return dict(aggregated_kpis)

    def _calculate_model_divergence(self) -> float:
        if not self.aggregated_edge_model_state or not self.global_model_from_cloud: return 0.0
        global_state = self.global_model_from_cloud.state_dict()
        global_keys = {k for k, p in global_state.items() if p.is_floating_point()}
        local_keys = {k for k, p in self.aggregated_edge_model_state.items() if p.is_floating_point()}
        common_keys = global_keys.intersection(local_keys)

        if not common_keys:
            self.logger.warning("Model divergence calculation: No common keys found between global and local models.")
            return 0.0

        divergence_tensors = [ torch.norm(self.aggregated_edge_model_state[k].cpu().float() - global_state[k].cpu().float())**2 for k in common_keys ]
        divergence = torch.sum(torch.stack(divergence_tensors))
        return torch.sqrt(divergence).item()
        
    def _recombine_model_parts(self, client_state, server_state, client_map, server_map):
        recombined = OrderedDict()
        for k_local, k_global in client_map.items():
            if k_local in client_state: recombined[k_global] = client_state[k_local].cpu().float()
        for k_local, k_global in server_map.items():
            if k_local in server_state: recombined[k_global] = server_state[k_local].cpu().float()
        full_model_template = get_unified_model_constructor(self.model_name, self.args.input_channels, self.args.output_channels)
        if set(recombined.keys()) == set(full_model_template.state_dict().keys()):
            return recombined
        self.logger.error(f"Recombined model keys mismatch. Missing: {set(full_model_template.state_dict().keys()) - set(recombined.keys())}")
        return None

    def average_state_dicts(self, state_dicts: List[dict], weights: Optional[List[float]] = None) -> Optional[dict]:
        if not state_dicts: return None
        
        if weights is None:
            weights = [1.0] * len(state_dicts)
        
        total_weight = sum(weights)
        if total_weight == 0: return None

        avg_dict = OrderedDict()
        for k in state_dicts[0].keys():
            weighted_tensors = [sd[k].cpu().float() * w for sd, w in zip(state_dicts, weights)]
            avg_dict[k] = torch.stack(weighted_tensors).sum(dim=0) / total_weight
        return avg_dict

    def shutdown(self):
        self.logger.info(f"Shutting down server for Edge {self.edge_id}...")
        if self.cloud_socket:
            try: self.cloud_socket.close()
            except Exception as e: self.logger.debug(f"Error closing cloud socket: {e}")
        self.disconnect_all_clients()
        self.stop_listening()
        self.logger.info(f"Server for Edge {self.edge_id} has been shut down.")
    
    def evaluate_on_local_validation_set(self) -> Tuple[float, float]:
     if not self.aggregated_edge_model_state or not self.val_loader: 
         return float('inf'), 0.0
         
     eval_model = get_unified_model_constructor(self.model_name, self.args.input_channels, self.args.output_channels)
     try:
         eval_model.load_state_dict(self.aggregated_edge_model_state)
     except RuntimeError as e:
         self.logger.error(f"Failed to load state dict for validation: {e}")
         return float('inf'), 0.0
         
     eval_model.to(self.device).eval()
     total_loss, num_batches, correct, total = 0.0, 0, 0, 0
     criterion = nn.CrossEntropyLoss()
     
     with torch.no_grad():
         for data, target in self.val_loader:
             data, target = data.to(self.device), target.to(self.device)
             output = eval_model(data)
             total_loss += criterion(output, target).item()
             
             pred = output.argmax(dim=1, keepdim=True)
             correct += pred.eq(target.view_as(pred)).sum().item()
             total += target.size(0)
             
             num_batches += 1
             
     avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
     accuracy = 100. * correct / total if total > 0 else 0.0
     
     self.logger.info(f"Edge {self.edge_id}: Evaluated on validation set. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
     
     return avg_loss, accuracy
    
    def evaluate_baseline_on_validation_set(self) -> float:
        if not self.global_model_from_cloud or not self.val_loader: return float('inf')
        self.global_model_from_cloud.eval()
        total_loss, num_batches = 0.0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model_from_cloud(data)
                total_loss += criterion(output, target).item()
                num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.logger.info(f"Edge {self.edge_id}: Evaluated baseline on validation set. Loss: {avg_loss:.4f}")
        self.baseline_validation_loss = avg_loss
        return avg_loss

    def safely_remove_client(self, client_id: int):
        with self.client_data_lock:
            if client_id in self.client_socket_dict:
                sock = self.client_socket_dict.pop(client_id)
                try: 
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except OSError:
                    pass
                except Exception as e:
                    self.logger.debug(f"Exception while closing client socket for client {client_id}: {e}")
            if client_id in self.client_raw_stats:
                del self.client_raw_stats[client_id]
            if client_id in self.current_mappo_decisions:
                del self.current_mappo_decisions[client_id]
            self.logger.info(f"Client {client_id} has been safely removed.")

    def _handle_client_handshake_thread(self, client_sock: socket.socket, client_addr):
        client_id_for_log = None
        try:
            msg_type, msg = self.recv_msg(client_sock, timeout=10.0)
            if msg_type == 'MSG_CLIENT_STATUS_TO_EDGE' and msg and 'client_id' in msg:
                cid = msg['client_id']
                client_id_for_log = cid
                with self.client_data_lock:
                    if cid in self.client_socket_dict:
                        self.logger.warning(f"Client {cid} is reconnecting. Overwriting its old socket entry.")
                    self.client_socket_dict[cid] = client_sock
                    self.client_raw_stats[cid] = msg.get('status_payload', {})
                if self.send_msg(client_sock, {'status': 'HANDSHAKE_OK'}, 'MSG_HANDSHAKE_OK'):
                    meta_msg_type, meta_msg = self.recv_msg(client_sock, timeout=10.0)
                    if meta_msg_type == 'MSG_CLIENT_METADATA' and meta_msg:
                        with self.client_data_lock:
                            if cid in self.client_raw_stats:
                                self.client_raw_stats[cid]['batches_per_epoch'] = meta_msg.get('batches_per_epoch', 0)
                    else:
                        raise ConnectionError("Failed to receive metadata from client after handshake ACK.")
                else:
                    raise ConnectionError("Failed to send handshake ACK to client.")
            else:
                self.logger.warning(f"Received invalid initial message from {client_addr} (type: {msg_type}). Closing connection.")
                client_sock.close()
        except (socket.timeout, ConnectionResetError, BrokenPipeError, EOFError) as e:
            log_level = self.logger.warning if isinstance(e, BrokenPipeError) else self.logger.error
            log_msg = f"Handshake process with {client_addr} (Client: {client_id_for_log}) failed due to {type(e).__name__}: {e}"
            log_level(log_msg, exc_info=False)
            if client_id_for_log is not None:
                self.safely_remove_client(client_id_for_log)
            try:
                if client_sock.fileno() != -1: client_sock.close()
            except Exception: pass
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during client handshake with {client_addr}: {e}", exc_info=True)
            if client_id_for_log is not None:
                self.safely_remove_client(client_id_for_log)
            try:
                if client_sock.fileno() != -1: client_sock.close()
            except Exception: pass