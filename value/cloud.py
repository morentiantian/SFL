import socket
from collections import OrderedDict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import time
import copy
import threading
import csv
import os
from typing import Dict, Any, Optional
from queue import Queue, Empty

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.sampling import get_dataset_cached
from models.Simple_split import get_unified_model_constructor
from utils.Communicator import Communicator
from utils.logger_setup import suppress_stdout

# ======================================================================================
# 1. Performance Logger (性能日志记录器) 
# ======================================================================================
class PerformanceLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_file_path = os.path.join(log_dir, f'performance_log_{experiment_name}.csv')
        self.fieldnames = [
            'global_round', 'test_accuracy', 'test_loss', 'round_time_s',
            'avg_computation_time_s', 'avg_communication_time_s',
            'avg_total_interaction_time_s', 'total_computation_energy_J',
            'total_communication_energy_J', 'total_energy_consumption_J',
            'avg_model_divergence'
        ]
        self._setup_log_file()

    def _setup_log_file(self):
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_round(self, round_data: Dict[str, Any]):
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            row_to_write = {field: round_data.get(field, 0) for field in self.fieldnames}
            writer.writerow(row_to_write)

# ======================================================================================
# 2. Heuristic-Weighted Averaging (HWA) Manager (启发式加权聚合管理器) - (与训练云对齐)
# ======================================================================================
class HWA_Manager:
    def __init__(self, args):
        self.args = args
        self.w_similarity = getattr(args, 'w_similarity', 1.0)
        self.w_data_size = getattr(args, 'w_data_size', 0.2)
        # 移除 self.alpha_anchor 和 self.anchor_model_state，锚点模型由CloudServer主逻辑直接管理
        self.logger = logging.getLogger("CloudProcess.HWA_Manager")
        self.logger.info(f"HWA Manager initialized with weights(sim={self.w_similarity}, data={self.w_data_size})")

    def _get_model_flat_tensor(self, model_state_dict: dict) -> torch.Tensor:
        # 这个函数本身没有问题，问题在于调用它的地方
        return torch.cat([p.cpu().view(-1) for p in model_state_dict.values() if p.is_floating_point()])

    def calculate_weights_and_aggregate(self, global_model_state: OrderedDict, edge_model_infos: list) -> OrderedDict:
        # ======================== 【核心修正】 ========================
        # 1. 在聚合前，先过滤掉那些没有有效模型数据的边缘节点报告
        payload_key = 'model_state' if self.args.test_policy != 'FedAvg_Baseline' else 'model_delta'
        valid_edge_model_infos = []
        for info in edge_model_infos:
            payload = info.get('payload', {})
            # 检查模型数据是否存在且不为空字典
            model_data = payload.get(payload_key)
            if model_data: 
                valid_edge_model_infos.append(info)
            else:
                self.logger.warning(f"HWA_Manager: Filtering out Edge {info.get('edge_id')} due to empty or missing model payload.")

        # 如果过滤后没有任何有效的模型，直接返回原始的全局模型
        if not valid_edge_model_infos:
            self.logger.warning("HWA_Manager: No valid edge models received for aggregation after filtering.")
            return global_model_state
        # =============================================================

        edge_ids = [info['edge_id'] for info in valid_edge_model_infos]
        payloads_map = {info['edge_id']: info['payload'] for info in valid_edge_model_infos}
        
        final_weights = {}
        total_weight = 0.0

        global_model_flat = self._get_model_flat_tensor(global_model_state)

        all_data_sizes = [payloads_map[edge_id].get('kpi_report', {}).get('total_data_size', 0) for edge_id in edge_ids]
        max_data_size = max(all_data_sizes) if all_data_sizes else 1
        if max_data_size == 0:
            max_data_size = 1

        for edge_id in edge_ids:
            model_info = payloads_map[edge_id]
            
            # 由于上面已经过滤，这里的 edge_model_state 不会是空的
            edge_model_state = model_info[payload_key]
            edge_model_flat = self._get_model_flat_tensor(edge_model_state)

            similarity_score = F.cosine_similarity(global_model_flat, edge_model_flat, dim=0).item()
            similarity_score = max(0, similarity_score)

            data_size = model_info.get('kpi_report', {}).get('total_data_size', 0)
            normalized_data_size = data_size / max_data_size
            
            weight = (self.w_similarity * similarity_score) + (self.w_data_size * normalized_data_size)
            final_weights[edge_id] = max(weight, 1e-6)
            total_weight += final_weights[edge_id]

        if total_weight > 0:
            for edge_id in final_weights: # 修正: 应该遍历 final_weights 的键
                final_weights[edge_id] /= total_weight
        else:
            self.logger.warning("Total weight is zero. Falling back to uniform averaging.")
            num_models = len(edge_ids)
            if num_models > 0:
                final_weights = {edge_id: 1.0 / num_models for edge_id in edge_ids}

        self.logger.info(f"HWA weights calculated: { {k: round(v, 3) for k, v in final_weights.items()} }")

        new_global_state = OrderedDict()
        target_device = next(iter(global_model_state.values())).device

        for key in global_model_state.keys():
            weighted_sum = torch.zeros_like(global_model_state[key], dtype=torch.float32, device=target_device)
            # 同样，只对有效的 final_weights 进行迭代
            for edge_id, weight in final_weights.items():
                payload = payloads_map.get(edge_id, {})
                if payload and payload_key in payload and key in payload[payload_key]:
                    weighted_sum += payload[payload_key][key].to(device=target_device, dtype=torch.float32) * weight
            new_global_state[key] = weighted_sum
            
        return new_global_state

# ======================================================================================
# 3. Heuristic Macro-Policy Coordinator (启发式宏观策略协调器) 
# ======================================================================================
class HeuristicPolicyCoordinator:
    def __init__(self, args):
        self.logger = logging.getLogger("CloudProcess.HeuristicCoordinator")
        self.args = args
        self.is_iid = (int(args.iid) == 1)

        self.mu = getattr(args, 'mu_initial')
        self.DIVERGENCE_TARGET = getattr(args, 'divergence_target')
        self.mu_min, self.mu_max = getattr(args, 'mu_min', 0.0), getattr(args, 'mu_max', 5)
        self.mu_adjust_rate = getattr(args, 'mu_adjust_rate', 1.2)
        self.logger.info(f"Coordinator initialized in Non-IID mode with dynamic mu.")
        self.logger.info(f"Initial mu = {self.mu}, Divergence Target = {self.DIVERGENCE_TARGET}")

    def update_macro_policy(self, aggregated_kpis: dict, current_global_round: int):
        if current_global_round < 10:
             return
             
        avg_model_divergence = aggregated_kpis.get('avg_model_divergence', 0.0)

        if avg_model_divergence > self.DIVERGENCE_TARGET:
            self.mu = min(self.mu * self.mu_adjust_rate, self.mu_max)
            self.logger.info(f"Divergence {avg_model_divergence:.3f} > target. Increasing mu to {self.mu:.4f}")
        else:
            self.mu = max(self.mu / self.mu_adjust_rate, self.mu_min)
            self.logger.info(f"Divergence {avg_model_divergence:.3f} <= target. Decreasing mu to {self.mu:.4f}")

    def get_policy_package(self) -> dict:
        return {'mu': self.mu, 'is_converged': False}

# ======================================================================================
# 4. Cloud Utility Agent (云端效用智能体)
# ======================================================================================
# 使用训练时的 CloudTimeSyncAgent 替换 CloudUtilityAgent
class CloudTimeSyncAgent:
    def __init__(self, args):
        self.args = args
        self.n_base = 3
        self.edge_time_cost_ema = {}
        self.ema_alpha = 0.3
        self.logger = logging.getLogger("CloudProcess.CloudTimeSyncAgent")
        self.logger.info("CloudTimeSyncAgent (efficiency-based) initialized for f_g_k decisions.")

    def get_all_actions(self, states_data: dict, active_edge_ids: list) -> dict:
        # 对于非MAPPO策略，返回固定值
        if self.args.test_policy not in ['HAC-SFL', 'MADRL-SFL']:
            return {eid: self.n_base for eid in active_edge_ids}
        
        if not states_data:
            return {edge_id: self.n_base for edge_id in active_edge_ids}

        time_costs = {}
        for edge_id in active_edge_ids:
            # T_unit 代表了该边缘节点的单位成本/时间
            current_cost = states_data.get(edge_id, {}).get('kpi_report', {}).get('T_unit', 10.0) + 1e-6
            
            # 使用指数移动平均 (EMA) 来平滑时间成本估计
            if edge_id not in self.edge_time_cost_ema:
                self.edge_time_cost_ema[edge_id] = current_cost
            else:
                self.edge_time_cost_ema[edge_id] = self.ema_alpha * current_cost + (1 - self.ema_alpha) * self.edge_time_cost_ema[edge_id]
            time_costs[edge_id] = self.edge_time_cost_ema[edge_id]
        
        # 将时间成本转化为速度得分（时间越短，速度越快，得分越高）
        speed_scores = {edge_id: 1.0 / cost for edge_id, cost in time_costs.items()}
        total_score = sum(speed_scores.values())
        
        if total_score == 0:
            return {edge_id: self.n_base for edge_id in active_edge_ids}
        
        # 计算总预算并按比例分配
        total_round_budget = len(active_edge_ids) * self.n_base
        allocations = {
            edge_id: max(1, min(int(round((score / total_score) * total_round_budget)), 8))
            for edge_id, score in speed_scores.items()
        }
        self.logger.info(f"[TimeSyncAgent] Decided f_g_k: {allocations}")
        return allocations

# ======================================================================================
# 5. 主云服务器 (Main Cloud Server) - (核心逻辑修改)
# ======================================================================================
class CloudServer(Communicator):
    def __init__(self, host, port, args):
        super().__init__()
        self.host, self.port, self.args = host, port, args
        self.expected_edges = args.num_edges
        self.global_round = 0
        
        self.logger = logging.getLogger("CloudProcess.CloudServer")
        
        # 使用与训练云一致的决策和聚合模块
        self.hwa_manager = HWA_Manager(args)
        self.policy_coordinator = HeuristicPolicyCoordinator(args)
        self.utility_agent = CloudTimeSyncAgent(args) # <-- 使用新的Agent
        
        self.perf_logger = PerformanceLogger(args.log_dir, args.experiment_name)
        
        self.model = get_unified_model_constructor(args.model_name, args.input_channels, args.output_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.model.to(self.device)
        self.logger.info(f"Cloud model '{args.model_name}' initialized on {self.device}")
        
        # 云端优化器和锚点模型逻辑与训练云对齐
        self.server_optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.logger.info("Initialized persistent ADAM optimizer on the cloud for server-side optimization.")
        self.anchor_model_state: Optional[OrderedDict] = copy.deepcopy(self.model.state_dict())
        
        # 学习率调度器逻辑保持不变，因为它与测试总轮次相关
        self.dummy_optimizer_for_scheduler = optim.SGD(self.model.parameters(), lr=self.args.sfl_lr)
        self.lr_scheduler = CosineAnnealingLR(self.dummy_optimizer_for_scheduler, T_max=args.epochs, eta_min=1e-6)
        self.logger.info(f"Initialized CosineAnnealingLR scheduler for client learning rate over {args.epochs} rounds.")

        with suppress_stdout():
            _, testset = get_dataset_cached(args.dataset, args.data_dir)
        self.v_test_loader = DataLoader(testset, batch_size=args.batch_size * 2, shuffle=False)
        
        self.edge_id_to_socket_map = {}
        self.edge_map_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_global_accuracy = 0.0
        self.logger.info("CloudServer components initialized.")

    def run(self, total_global_rounds):
        self.logger.info(f"[Cloud] Starting SFL test process for {total_global_rounds} rounds.")
        edge_states = {}

        for self.global_round in range(total_global_rounds):
            if self.stop_event.is_set(): break

            self.logger.info(f"\n[Cloud] === Starting Global Round {self.global_round + 1}/{total_global_rounds} ===")
            round_start_time = time.time()
            
            with self.edge_map_lock:
                active_edge_ids = list(self.edge_id_to_socket_map.keys())
            
            if not active_edge_ids:
                self.logger.warning("[Cloud] No active edges. Waiting...")
                time.sleep(5); continue

            avg_loss, current_test_acc = self.evaluate_on_test()
            
            global_accuracy_gain = current_test_acc - self.last_global_accuracy
            self.last_global_accuracy = current_test_acc
            
            self.lr_scheduler.step()
            current_sfl_lr_for_clients = self.lr_scheduler.get_last_lr()[0]
            
            aggregated_kpis = self._aggregate_edge_kpis(edge_states)
            self.policy_coordinator.update_macro_policy(aggregated_kpis, self.global_round)
            
            macro_policy_package = self.policy_coordinator.get_policy_package()
            macro_policy_package['global_accuracy_gain'] = global_accuracy_gain
            macro_policy_package['current_sfl_lr'] = current_sfl_lr_for_clients
            self.logger.info(f"[Cloud] Current SFL learning rate for clients is {current_sfl_lr_for_clients:.6f}")
            
            # 调用新的 f_g_k 决策 Agent
            decided_local_rounds_map = self.utility_agent.get_all_actions(edge_states, active_edge_ids)

            received_payloads, received_states = self._distribute_and_collect_robustly(decided_local_rounds_map, macro_policy_package)
            edge_states = received_states

            if received_payloads:
                # 统一聚合和更新逻辑，与训练云对齐
                if self.args.test_policy == 'FedAvg_Baseline':
                    aggregated_global_delta = self._aggregate_edge_deltas(received_payloads)
                    if aggregated_global_delta:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in aggregated_global_delta:
                                    param.data += aggregated_global_delta[name].to(self.device)
                else:
                    agg_state = self.hwa_manager.calculate_weights_and_aggregate(
                        self.model.state_dict(), received_payloads
                    )
                    if agg_state:
                        # 直接更新锚点模型，不再使用EMA
                        self.anchor_model_state = copy.deepcopy(agg_state)
                        self.logger.info("Anchor model updated with the latest aggregated state.")

                        pseudo_gradient = OrderedDict()
                        current_global_state = self.model.state_dict()
                        for key in current_global_state:
                            if key in agg_state and current_global_state[key].is_floating_point():
                                pseudo_gradient[key] = current_global_state[key].detach() - agg_state[key].to(self.device)
                        
                        self.server_optimizer.zero_grad()
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and name in pseudo_gradient:
                                param.grad = pseudo_gradient[name]
                        
                        self.server_optimizer.step()
                        self.logger.info("Global model updated using server-side ADAM optimizer.")
                    else:
                        self.logger.warning("Aggregation resulted in an empty state. Global model not updated.")
            else:
                self.logger.warning(f"No models or deltas received in round {self.global_round + 1}. Global model not updated.")

            round_duration = time.time() - round_start_time
            round_perf_data = self._prepare_performance_log_data(current_test_acc, avg_loss, round_duration, received_states)
            self.perf_logger.log_round(round_perf_data)
            self.logger.info(f"[Cloud] Round {self.global_round + 1} finished. Time: {round_duration:.2f}s, Test Acc: {current_test_acc:.2f}%")
            
            self._broadcast_round_complete_signal()
            
        self.logger.info("[Cloud] Global training loop finished.")
        self.close()

    def _distribute_and_collect_robustly(self, decided_local_rounds_map: dict, macro_policy_package: dict):
        self.logger.info(f"[Cloud] Distributing tasks to {len(decided_local_rounds_map)} edges: {list(decided_local_rounds_map.keys())}")
        current_global_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        threads = []
        results_queue = Queue()

        def _collect_from_edge(edge_id, sock):
            try:
                msg_type, response = self.recv_msg(sock, timeout=600.0) 
                if msg_type == 'MSG_LOCAL_MODEL_TO_CLOUD' and response:
                    results_queue.put({'edge_id': edge_id, 'data': response, 'status': 'success'})
                else:
                    raise ConnectionError(f"Received invalid response (type: {msg_type})")
            except Exception as e:
                self.logger.error(f"Failed to receive report from Edge {edge_id} due to: {e}. It will be marked as failed for this round.")
                results_queue.put({'edge_id': edge_id, 'data': None, 'status': 'failed'})

        with self.edge_map_lock:
            edges_in_this_round = []
            
            for edge_id in list(self.edge_id_to_socket_map.keys()):
                sock = self.edge_id_to_socket_map.get(edge_id)
                f_g_k = decided_local_rounds_map.get(edge_id)
                if not sock or f_g_k is None:
                    continue

                #  CloudServer (self) 获取 anchor_model_state
                anchor_state_for_edge = self.anchor_model_state
                
                message_to_edge = {
                    'model_state_dict': current_global_state,
                    'edge_rounds': f_g_k,
                    'current_global_round': self.global_round,
                    'anchor_model_state': anchor_state_for_edge,
                    'macro_policy_package': macro_policy_package,
                    'sfl_lr': macro_policy_package.get('current_sfl_lr')
                }

                if self.send_msg(sock, message_to_edge, 'MSG_CLOUD_DECISION_TO_EDGE'):
                    thread = threading.Thread(target=_collect_from_edge, args=(edge_id, sock))
                    threads.append((thread, edge_id)) 
                    edges_in_this_round.append(edge_id) 
                    thread.start()
                else:
                    self.logger.error(f"Failed to send task to Edge {edge_id}. It will be removed.")
                    self._handle_disconnected_edge(edge_id)
        
        for thread, edge_id in threads:
            thread.join()
        
        received_payloads, received_states = [], {}
        successful_edges = set()
        while not results_queue.empty():
            result = results_queue.get()
            edge_id = result['edge_id']
            if result['status'] == 'success':
                successful_edges.add(edge_id)
                response = result['data']
                if 'model_payload' in response and response['model_payload']:
                    received_payloads.append({'edge_id': edge_id, 'payload': response['model_payload']})
                if 'next_edge_state' in response:
                    received_states[edge_id] = response['next_edge_state']
        
        failed_edges = set(edges_in_this_round) - successful_edges
        if failed_edges:
            self.logger.warning(f"The following edges failed to report back this round: {list(failed_edges)}. They will be removed.")
            for edge_id in failed_edges:
                self._handle_disconnected_edge(edge_id, "Did not report back successfully.")

        return received_payloads, received_states

    def _broadcast_round_complete_signal(self):
        self.logger.info("[Cloud] Broadcasting 'ROUND_COMPLETE' signal to all edges...")
        with self.edge_map_lock:
            edge_sockets = list(self.edge_id_to_socket_map.items())

        for edge_id, sock in edge_sockets:
            try:
                if not self.send_msg(sock, {'status': 'ROUND_COMPLETE'}, 'MSG_CLOUD_ROUND_COMPLETE'):
                    self.logger.warning(f"Failed to send ROUND_COMPLETE signal to Edge {edge_id}. It might have disconnected.")
            except Exception as e:
                self.logger.error(f"Error sending ROUND_COMPLETE signal to Edge {edge_id}: {e}")

    def _aggregate_edge_deltas(self, edge_model_infos: list) -> OrderedDict:
        if not edge_model_infos:
            return OrderedDict()

        valid_infos = [info for info in edge_model_infos if info['payload'].get('model_delta')]
        if not valid_infos:
            return OrderedDict()

        num_edges = len(valid_infos)
        aggregated_delta = OrderedDict((k, torch.zeros_like(v)) for k, v in valid_infos[0]['payload']['model_delta'].items())

        for info in valid_infos:
            edge_delta = info['payload']['model_delta']
            for key in aggregated_delta:
                if key in edge_delta:
                    aggregated_delta[key] += edge_delta[key] / num_edges
                    
        self.logger.info(f"Successfully aggregated edge deltas from {num_edges} edges.")
        return aggregated_delta


    def _handle_disconnected_edge(self, edge_id, error_msg=""):
        with self.edge_map_lock:
            if edge_id in self.edge_id_to_socket_map:
                self.logger.warning(f"Edge {edge_id} is disconnected. {error_msg}. Removing it from the active pool.")
                sock = self.edge_id_to_socket_map.pop(edge_id)
                try: 
                    sock.close()
                except Exception as e: 
                    self.logger.debug(f"Error closing socket for disconnected edge {edge_id}: {e}")

    def accept_edges_initial(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.expected_edges * 2)
        self.logger.info(f"[Cloud] Listening on {self.host}:{self.port} for {self.expected_edges} edges...")
        
        while len(self.edge_id_to_socket_map) < self.expected_edges:
            if self.stop_event.is_set(): return
            try:
                conn, addr = self.server_sock.accept()
                conn.settimeout(300.0)
                msg_type, hello_msg = self.recv_msg(conn)
                if msg_type == 'MSG_EDGE_HELLO' and hello_msg and 'edge_id' in hello_msg:
                    edge_id = hello_msg['edge_id']
                    with self.edge_map_lock:
                        self.edge_id_to_socket_map[edge_id] = conn
                    self.logger.info(f"[Cloud] Handshake successful with Edge {edge_id} from {addr}.")
                else: 
                    self.logger.warning(f"Received invalid hello message from {addr}. Closing connection.")
                    conn.close()
            except Exception as e:
                if not self.stop_event.is_set():
                    self.logger.error(f"[Cloud] Error accepting edge: {e}", exc_info=True)
        self.logger.info(f"[Cloud] All {self.expected_edges} edges connected.")
    
    def _prepare_performance_log_data(self, test_acc, test_loss, round_time, edge_states: dict) -> Dict[str, Any]:
        log_data = {'global_round': self.global_round + 1, 'test_accuracy': test_acc, 'test_loss': test_loss, 'round_time_s': round_time}
        kpi_reports = [state.get('kpi_report', {}) for state in edge_states.values()]
        if kpi_reports:
            for key in self.perf_logger.fieldnames[4:]:
                if key.startswith('total_'):
                    log_data[key] = np.sum([r.get(key, 0) for r in kpi_reports])
                elif key.startswith('avg_'):
                    valid_reports = [r.get(key) for r in kpi_reports if r.get(key) is not None]
                    if valid_reports:
                        log_data[key] = np.mean(valid_reports)
        return log_data

    def evaluate_on_test(self):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        if not self.v_test_loader: return 0.0, 0.0
        with torch.no_grad():
            for data, target in self.v_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss_sum += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = loss_sum / total if total > 0 else float('inf')
        acc = 100. * correct / total if total > 0 else 0
        self.logger.info(f"[Cloud][Eval] Round {self.global_round + 1} - Test Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        return avg_loss, acc

    def _aggregate_edge_kpis(self, edge_states: dict) -> dict:
        if not edge_states: return {}
        divergences = [state.get('kpi_report', {}).get('model_divergence', 0) for state in edge_states.values()]
        return {'avg_model_divergence': np.mean(divergences) if divergences else 0}
    
    def broadcast_stop_signal(self):
        self.logger.info("[Cloud] Broadcasting stop signal to all edges.")
        with self.edge_map_lock:
            for sock in self.edge_id_to_socket_map.values():
                try: self.send_msg(sock, {'status': 'STOP_TRAINING'}, 'MSG_CLOUD_STOP')
                except Exception: pass

    def close(self):
        self.logger.info("Shutting down Cloud Server...")
        self.stop_event.set()
        if hasattr(self, 'server_sock'):
            try: self.server_sock.close()
            except Exception: pass
        self.broadcast_stop_signal()
        time.sleep(1) 
        with self.edge_map_lock:
            for sock in self.edge_id_to_socket_map.values():
                try: sock.close()
                except Exception: pass
            self.edge_id_to_socket_map.clear()