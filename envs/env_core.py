import logging
import os
import sys
from typing import List, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random

from datasets.sampling import prepare_validation_loader
from value.edge import EdgeServer 
from models.Simple_split import get_model_structural_features_for_all_splits, get_unified_num_split_options

class EnvCore(gym.Env):
    metadata = {'render_modes': ['human'], 'name': "sfl_uav_mappo_env_v2_aligned"}

    def __init__(self, args, rank: int = 0):
        super(EnvCore, self).__init__()
        self.args = args
        self.server_id = rank
        args.edge_id = self.server_id
        self.num_agents = args.num_agents
        self.episode_length = args.epochs 
        self.current_step_in_episode = 0

        self.logger = logging.getLogger(f"EdgeRunner-{self.server_id}.EnvCore")
        
        val_loader = prepare_validation_loader(args, self.server_id, self.logger)
        self.server = EdgeServer(args=args, logger_ref=self.logger, val_loader=val_loader)

        # 为奖励函数初始化状态变量
        self.best_accuracy_so_far = 0.0
        self.last_round_accuracy = 0.0
        
        # 定义奖励权重
        self.reward_weights = {
            'w_perf': getattr(args, 'w_perf', 1.0),
            'w_time': getattr(args, 'w_time', 0.1),
            'w_cost': getattr(args, 'w_cost', 0.2),
        }

        try:
            cloud_ip = '127.0.0.1'
            cloud_port = getattr(args, 'cloud_port', 9000)
            if not self.server.connect_to_cloud(cloud_ip, cloud_port):
                raise RuntimeError("Failed to connect to Cloud server during initialization.")
            self.logger.info(f"Edge server {self.server_id} has successfully connected to the cloud.")
        except Exception as e:
            self.logger.error(f"Network setup for cloud connection failed: {e}", exc_info=True)
            raise
        
        # --- MAPPO 智能体与空间定义  ---
        example_input_tensor = self._get_example_tensor(args)
        model_features_info = get_model_structural_features_for_all_splits(args.model_name, example_input_tensor, args.output_channels, self.logger)
        self.model_structural_features_vector = model_features_info["features_vector"]
        
        dim_uav_pure_local_features = 8
        # 云端策略向量维度
        dim_cloud_policy_features = 1 + 3 # mu + 3 reward weights (perf, time, cost)
        self.obs_dim_per_agent = dim_uav_pure_local_features + len(self.model_structural_features_vector) + dim_cloud_policy_features + 1

        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim_per_agent,), dtype=np.float32) for _ in range(self.num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents * self.obs_dim_per_agent,), dtype=np.float32) for _ in range(self.num_agents)]
        
        cut_dim = get_unified_num_split_options(args.model_name, self.logger)
        # 本地迭代次数选项
        self.local_iteration_options = [4, 8, 16, 24, 32, 64]
        edge_iter_dim = len(self.local_iteration_options)
        
        self.action_space = [spaces.MultiDiscrete([cut_dim, edge_iter_dim]) for _ in range(self.num_agents)]
        self.logger.info(f"EnvCore (Test Aligned) Initialized. Obs dim: {self.obs_dim_per_agent}.")

    def set_mappo_policy(self, policy):
        self.server.policy = policy
        self.logger.info("Pre-trained MAPPO policy has been injected into the edge server.")

    def _get_example_tensor(self, args):
        if args.dataset in ['cifar10', 'cifar100']:
            return torch.randn(1, args.input_channels, 32, 32)
        else:
            raise ValueError(f"Unsupported dataset for example tensor: {args.dataset}")

    def _get_obs(self) -> Tuple[List[np.ndarray], np.ndarray]:
        individual_obs_list = []
        active_masks = np.zeros((self.num_agents, 1), dtype=np.float32)
        
        mapped_uav_ids = self.server.selected_clients_in_order
        cloud_policy_package = self.server.macro_policy_package_from_cloud
        
        # 使用固定的基础权重来构建观测空间，确保其维度一致性
        reward_weights = self.reward_weights
        cloud_policy_vector = np.array([
            cloud_policy_package.get('mu', 0.1),
            reward_weights.get('w_perf', 1.0),
            reward_weights.get('w_time', 0.1),
            reward_weights.get('w_cost', 0.2)
        ], dtype=np.float32)

        normalized_training_stage = self.current_step_in_episode / self.episode_length

        for agent_slot_idx in range(self.num_agents):
            client_id = mapped_uav_ids[agent_slot_idx] if agent_slot_idx < len(mapped_uav_ids) else None
            if client_id is not None and client_id in self.server.client_raw_stats:
                client_features = self.server.get_client_features_for_mappo(client_id)
                combined_obs = np.concatenate([
                    client_features, 
                    self.model_structural_features_vector, 
                    cloud_policy_vector,
                    np.array([normalized_training_stage])
                ]).astype(np.float32)
                individual_obs_list.append(combined_obs)
                active_masks[agent_slot_idx, 0] = 1.0
            else:
                individual_obs_list.append(np.zeros(self.obs_dim_per_agent, dtype=np.float32))
                
        return individual_obs_list, active_masks

    def reset(self) -> List[np.ndarray]:
        # Reset 仅在测试开始时由 runner 调用一次
        self.logger.info(f"\n[EnvCore] === Resetting for a new Test Run (Edge ID: {self.server_id}) ===\n")
        self.current_step_in_episode = 0
        
        # 在这里执行一次性的客户端连接
        num_clients_per_edge = self.args.num_clients // self.args.num_edges
        self.server.accept_persistent_connections(
            num_expected_clients=num_clients_per_edge,
            timeout=60  # 可以给一个稍长的时间确保所有客户端进程启动并连接
        )
        
        # 重置奖励函数所需的状态
        self.best_accuracy_so_far = 0.0
        self.last_round_accuracy = 0.0
        
        # 在这里执行一次 map_clients_to_agent_slots 来获取初始状态
        self.server.map_clients_to_agent_slots(self.args.num_agents)
        initial_obs_list, _ = self._get_obs()
        return initial_obs_list
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[List[float]], List[bool], List[dict]]:
        self.current_step_in_episode += 1

        try:
            self.server._request_and_update_all_client_stats()

            self.server.map_clients_to_agent_slots(self.args.num_agents)
            
            self.logger.info(f"\n[EnvCore Step {self.current_step_in_episode}] Starting SFL training with persistently connected clients...")
            
            parsed_actions = {}
            selected_client_ids = self.server.selected_clients_in_order
            for i, client_id in enumerate(selected_client_ids):
                if client_id is not None and i < len(actions) and actions[i] is not None:
                    split_layer_index = int(actions[i][0])
                    local_iterations_index = int(actions[i][1])
                    parsed_actions[client_id] = {
                        'split_layer': split_layer_index + 1,
                        'local_iterations': self.local_iteration_options[local_iterations_index]
                    }
            self.server.current_mappo_decisions = parsed_actions

            success, _ = self.server.receive_global_config_from_cloud()
            if not success: raise ConnectionError("Failed to receive configuration from cloud.")
            
            self.server.evaluate_baseline_on_validation_set()
            exec_results = self.server.execute_all_sfl_interactions_for_global_round()
            reward_val = self.calculate_aligned_mappo_reward(exec_results, actions)
            
            if not self.server.finalize_and_report_to_cloud():
                self.logger.warning("[EnvCore step] Failed to report final results to cloud.")

            self.server.wait_for_round_completion_signal()

            next_obs_list, active_masks = self._get_obs()
            done = (self.current_step_in_episode >= self.episode_length)
            
            rewards_list = [[reward_val] if active_masks[i, 0] > 0 else [0.0] for i in range(self.num_agents)]
            return next_obs_list, rewards_list, [done] * self.num_agents, [{'active_masks': active_masks}] * self.num_agents

        except Exception as e:
            self.logger.error(f"[EnvCore step CRITICAL] Unhandled exception: {e}", exc_info=True)
            obs_list, active_masks = self._get_obs()
            rewards_list = [[-10.0] for _ in range(self.num_agents)]
            dones = [True] * self.num_agents
            infos = [{'active_masks': active_masks}] * self.num_agents
            return obs_list, rewards_list, dones, infos

    def calculate_aligned_mappo_reward(self, exec_results: Dict, actions: List[np.ndarray]) -> float:
        if not exec_results or exec_results.get('status') != 'success':
            self.logger.warning("[Reward] SFL execution failed, returning heavy penalty.")
            return -10.0

        final_loss, final_accuracy = self.server.evaluate_on_local_validation_set()
        base_loss = self.server.baseline_validation_loss
        sfl_round_time = exec_results.get('total_time_s', 120.0)

        if np.isinf(final_loss) or np.isnan(final_loss) or final_loss <= 0:
            self.logger.warning(f"[Reward] Encountered invalid loss: {final_loss}, returning heavy penalty.")
            return -10.0

        # (1) 准确率提升速度奖励 (Velocity Reward)
        accuracy_gain = final_accuracy - self.last_round_accuracy
        velocity_reward = 0.0
        if accuracy_gain > 0.01: # 只有在有明显提升时才给予奖励
            velocity_reward = accuracy_gain * 100 

        # (2) 平台期突破奖励 (Plateau Bonus)
        plateau_bonus = 0.0
        if final_accuracy > self.best_accuracy_so_far:
            plateau_bonus = 50.0  
            self.best_accuracy_so_far = final_accuracy
        
        # (3) 损失下降奖励 (Loss Reduction Reward)
        loss_reduction = base_loss - final_loss
        loss_reward = 0.0
        if accuracy_gain <= 0.01 and loss_reduction > 0:
            loss_reward = loss_reduction * 20

        # (4) 准确率回退惩罚 (Accuracy Penalty)
        accuracy_penalty = 0.0
        if accuracy_gain < -0.5: # 仅在显著下降时惩罚
            accuracy_penalty = abs(accuracy_gain) * 30
        
        # (5) 成本惩罚 (Cost Penalty)
        time_penalty = 0.0
        cost_penalty = 0.0
        if final_accuracy > 20.0: # 在模型具备一定性能后才开始考虑成本
            MAX_REASONABLE_TIME = 120.0
            time_penalty = (sfl_round_time / MAX_REASONABLE_TIME) * self.reward_weights['w_time']
            
            num_decision_agents = len(self.server.current_mappo_decisions)
            if num_decision_agents > 0:
                max_split_idx = self.action_space[0].nvec[0] - 1
                max_epoch_idx = len(self.local_iteration_options) - 1
                total_action_cost = 0.0
                for i, uav_id in enumerate(self.server.selected_clients_in_order):
                    if uav_id in self.server.current_mappo_decisions:
                        norm_split_cost = actions[i][0] / max_split_idx if max_split_idx > 0 else 0
                        norm_epoch_cost = actions[i][1] / max_epoch_idx if max_epoch_idx > 0 else 0
                        total_action_cost += (0.4 * norm_split_cost + 0.6 * norm_epoch_cost)
                avg_action_cost = total_action_cost / num_decision_agents
                cost_penalty = self.reward_weights['w_cost'] * avg_action_cost

        final_reward = (
            velocity_reward + 
            plateau_bonus + 
            loss_reward -
            accuracy_penalty -
            time_penalty - 
            cost_penalty
        )

        # 更新状态以备下轮使用
        self.last_round_accuracy = final_accuracy

        self.logger.info(
            f"[Step]: {self.current_step_in_episode} "
            f"[Reward] Final: {final_reward:.4f} (Velo: {velocity_reward:.2f}, Bonus: {plateau_bonus:.2f}, "
            f"LossRwd: {loss_reward:.4f}, AccPenalty: {-accuracy_penalty:.2f}, " 
            f"TimePenalty: {-time_penalty:.4f}, CostPenalty: {-cost_penalty:.4f}) | "
            f"Loss: {final_loss:.4f} | Acc: {final_accuracy:.2f}% (Best: {self.best_accuracy_so_far:.2f}%)"
        )

        return float(final_reward)

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            self.logger.info(f"Env random seed set to: {seed}")

    def render(self, mode='human'):
        pass

    def close(self):
        # 在环境关闭时，确保所有持久连接被断开
        if hasattr(self.server, 'disconnect_all_clients'):
            self.logger.info(f"Closing environment and disconnecting all persistent clients for Edge {self.server_id}.")
            self.server.disconnect_all_clients()
            
        if hasattr(self.server, 'shutdown'):
            self.server.shutdown()
        self.logger.info(f"Environment for Edge {self.server_id} has been fully closed.")