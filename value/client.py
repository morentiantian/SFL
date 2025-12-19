import itertools
import random
import logging
import socket
import time
import copy
from collections import OrderedDict, Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, Dataset

from utils.Communicator import Communicator, connect_with_retry
from models.Simple_split import get_unified_model_constructor, get_unified_split_model_function
from utils.logger_setup import suppress_stdout

# --- 定义用于模拟的常量 ---
MAX_DATASET_SIZE = 5000
MAX_CPU_FREQUENCY = 3.5  # GHz
MAX_CHANNEL_QUALITY = 500.0 # 信噪比指标
MAX_BANDWIDTH = 30   # MHz
MAX_DISTANCE = 500 # 米
GAMMA_CPU = 0.5
ETA_CHANNEL = 0.5
LAMBDA_DIST = 0.01
TX_POWER_WATTS = 0.25
RX_POWER_WATTS = 0.1
COMP_POWER_COEFF = 2.0
BYTES_PER_FLOAT32 = 4

class Client(Communicator):
    def __init__(self, args, local_data_indices: List[int]):
        super().__init__()
        self.args = args
        self.id = args.client_id
        self.logger = logging.getLogger(f"Client_{self.id}")

        # 根据客户端ID和总客户端数/边缘节点数，计算它唯一归属的边缘服务器ID
        num_clients_per_edge = self.args.num_clients // self.args.num_edges
        self.my_edge_id = self.id // num_clients_per_edge

        # 客户端只连接属于自己的那个边缘服务器
        self.server_port_base = getattr(args, 'server_port_base', 7000)
        
        # 将原来的多地址列表改为单一地址的目标
        target_edge_address = ('127.0.0.1', self.server_port_base + self.my_edge_id)
        self.edge_server_addresses = [target_edge_address]
        # --------------------

        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        # --- 模型与优化器 ---
        self.model: Optional[nn.Module] = None
        self.anchor_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # 同一个“大回合”内的多次边缘交互中，保持优化器的状态（如动量）
        self.optimizer_state_for_round: Optional[Dict[str, Any]] = None
        self.round_of_last_optimizer_state: int = -1
        
        # --- 数据管理 ---
        self.full_train_dataset_ref: Optional[Dataset] = None
        self.current_dataloader: Optional[DataLoader] = None
        self.full_local_data_indices = local_data_indices
        assert len(self.full_local_data_indices) >= 128

        # --- SFL 状态 ---
        self.sock = None
        self.split_point: Optional[int] = None
        self.local_iterations: int = 0
        self.current_global_round: int = 0
        self.last_round_participated: int = -1
        
        # --- 动态状态变量 ---
        self.battery_level = round(random.uniform(0.7, 1.0), 2)
        self.cpu_frequency = 0.0
        self.channel_quality = 0.0
        self.bandwidth = 0
        self.distance = 0.0
        self.U: float = 0.0
        
        # [MHR-SFL] 存储上一轮分配的资源，用于计算本轮Utility
        self.last_allocated_comp = None
        self.last_allocated_bw = None

        # --- 客户端质量等级 ---
        num_clients = args.num_clients
        if self.id < int(num_clients * 0.4):
            self.quality_tier = 'low'
        elif self.id < int(num_clients * 0.8):
            self.quality_tier = 'medium'
        else:
            self.quality_tier = 'high'
        self.logger.info(f"Client {self.id} initialized as a '{self.quality_tier}' quality client, assigned to Edge {self.my_edge_id} on {self.device}.")

    def prepare_dataloader(self) -> bool:
        # 如果内存中没有数据集的引用，则从磁盘加载
        if self.full_train_dataset_ref is None:
            self.logger.info("Dataset reference not found, loading dataset locally...")
            try:
                from datasets.sampling import get_dataset_cached
                with suppress_stdout():
                    self.full_train_dataset_ref, _ = get_dataset_cached(self.args.dataset, self.args.data_dir)
            except Exception as e:
                self.logger.error(f"Failed to load dataset locally: {e}", exc_info=True)
                return False
        
        if not self.full_local_data_indices:
            self.logger.warning("Client has no data indices. Training will be skipped.")
            return False

        # 根据分配给此客户端的数据索引，创建数据子集和加载器
        current_round_subset = Subset(self.full_train_dataset_ref, self.full_local_data_indices)
        self.current_dataloader = DataLoader(current_round_subset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        if len(self.current_dataloader) == 0:
            self.logger.warning("DataLoader is empty for this client. Training will be skipped.")
            return False
        return True

    def start(self):
        # 客户端进程的主循环 -> 修改为一次性连接循环
        self.logger.info(f"Client {self.id} process started. Attempting persistent connection to Edge {self.my_edge_id}.")
        
        try:
            # 1. 尝试连接到其指定的边缘服务器
            self.sock = connect_with_retry(
                self.edge_server_addresses, 
                logger_ref=self.logger,
                initial_delay=2, max_delay=10, max_retries=10
            )
            
            if not self.sock:
                self.logger.critical("Failed to establish a persistent connection to the edge server. Client is terminating.")
                return

            # 2. 执行一次性握手
            if not self._perform_handshake():
                self.logger.critical("Handshake with server failed. Client is terminating.")
                self.close_connection()
                return 
            
            self.logger.info(f"Persistent connection established with {self.sock.getpeername()}. Standing by for tasks for the entire episode.")
            
            # 3. 进入一个永久的监听循环
            while True:
                idle_start_time = time.perf_counter()
                
                msg_type, directive = self.recv_msg(self.sock, timeout=None) 

                if msg_type is None and directive is None:
                    self.logger.info("Connection gracefully closed by server, likely end of experiment. Client is shutting down.")
                    break

                # 增加一个新的分支，用于响应服务器的状态更新请求
                if msg_type == 'MSG_REQUEST_STATUS_UPDATE':
                    self.logger.debug("Received status update request from edge server.")
                    # 获取最新的状态负载
                    status_payload = self._update_and_get_client_status()
                    # 将其发送回服务器
                    self.send_msg(self.sock, {'status_payload': status_payload}, 'MSG_CLIENT_STATUS_UPDATE')

                elif msg_type == 'MSG_TRAINING_DIRECTIVE_TO_CLIENT':
                    idle_time_s = time.perf_counter() - idle_start_time
                    
                    if directive.get('participate', False):
                        self.current_global_round = directive.get('current_global_round', self.current_global_round)
                        training_mode = directive.get('training_mode', 'sfl')
                        self.logger.info(f"Received task for GR {self.current_global_round + 1} in {training_mode.upper()} mode after {idle_time_s:.2f}s of idle time.")
                        
                        if training_mode == 'fedavg':
                            self._handle_fedavg_interaction(directive, idle_time_s)
                        else:
                            self._handle_sfl_interaction(directive, idle_time_s)
                    else:
                        self.logger.info(f"Received standby directive for GR {directive.get('current_global_round', -1) + 1}.")
                
                elif msg_type == 'MSG_CLOUD_ROUND_COMPLETE' and directive.get('status') == 'EXPERIMENT_COMPLETE':
                    self.logger.info("Received experiment completion signal from server. Shutting down.")
                    break
                else:
                    self.logger.warning(f"Received unknown or unexpected message type: {msg_type}.")
        
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self.logger.error(f"Persistent connection lost: {e}. Client will terminate.", exc_info=False)
        except Exception as e:
            self.logger.critical(f"An unexpected fatal error occurred in the main loop, terminating client: {e}", exc_info=True)
        finally:
            self.close_connection()
            self.logger.info(f"Client {self.id} process has terminated.")
            
    def _perform_handshake(self) -> bool:
        # 客户端执行三步握手协议
        if not self._send_status():
            self.logger.error("Failed to send initial status during handshake.")
            return False
        
        self.logger.debug("Status sent. Waiting for handshake acknowledgement from server...")
        
        msg_type, _ = self.recv_msg(self.sock, timeout=60.0)
        if msg_type != 'MSG_HANDSHAKE_OK':
            self.logger.warning(f"Handshake failed. Expected 'MSG_HANDSHAKE_OK', but received '{msg_type}'.")
            return False

        if not self.prepare_dataloader():
            self.logger.error("Failed to prepare dataloader after handshake ACK.")
            return False

        batches_per_epoch = len(self.current_dataloader)
        if not self.send_msg(self.sock, {'batches_per_epoch': batches_per_epoch}, 'MSG_CLIENT_METADATA'):
            self.logger.error("Failed to send client metadata (batches_per_epoch).")
            return False
        
        return True

    def _handle_sfl_interaction(self, directive: dict, idle_time_s: float = 0.0):
        # 处理一次完整的SFL交互
        if not self.prepare_dataloader():
            self.logger.error("Failed to prepare dataloader. Aborting SFL interaction.")
            return
        if not self._setup_for_sfl_training(directive):
            self.logger.error("Failed to set up for SFL training. Aborting SFL interaction.")
            return

        training_success, metrics = self._execute_local_sfl_training()
        if training_success:
            metrics['idle_time_s'] = idle_time_s 
            self._upload_final_sfl_report(metrics)
            self.last_round_participated = self.current_global_round
        else:
            self.logger.error("Local SFL training failed. Report not sent.")
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _handle_fedavg_interaction(self, directive: dict, idle_time_s: float = 0.0):
        # 处理一次完整的FedAvg交互
        self.logger.info(f"==> Starting FedAvg interaction for GR {self.current_global_round + 1}. <==")
        if not self.prepare_dataloader():
            self.logger.error("Failed to prepare dataloader for FedAvg. Aborting.")
            return
        
        try:
            # 每次FedAvg交互都使用全新的模型和优化器
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            self.optimizer = None
            torch.cuda.empty_cache()

            self.model = get_unified_model_constructor(
                self.args.model_name, self.args.input_channels, self.args.output_channels
            )
            self.model.load_state_dict(directive['model_state_dict'])
            self.model.to(self.device)

            initial_model_state = {k: v.cpu().detach() for k, v in self.model.state_dict().items()}
            learning_rate = directive.get('sfl_lr', self.args.sfl_lr)
            self.logger.info(f"Creating new SGD optimizer for FedAvg with LR: {learning_rate:.6f}")
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
                
        except Exception as e:
            self.logger.error(f"Failed to set up model for FedAvg: {e}", exc_info=True)
            return
        
        self.local_iterations = directive.get('local_iterations', 10)
        training_success, metrics = self._execute_local_fedavg_training(self.local_iterations, self.optimizer)
        
        if training_success:
            final_model_state = {k: v.cpu().detach() for k, v in self.model.state_dict().items()}
            model_delta = OrderedDict([(key, final_model_state[key] - initial_model_state[key]) for key in final_model_state])

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            metrics['idle_time_s'] = idle_time_s
            comp_energy, comm_energy, total_energy = self._simulate_energy_fedavg(metrics['computation_time_s'])
            
            kpi_report = {
                'total_data_size': len(self.full_local_data_indices),
                'computation_time_s': metrics['computation_time_s'],
                'communication_time_s': metrics['communication_time_s'],
                'total_interaction_time_s': metrics['total_interaction_time_s'],
                'idle_time_s': metrics['idle_time_s'], # 将新指标加入报告
                'computation_energy_J': comp_energy,
                'communication_energy_J': comm_energy,
                'total_energy_consumption_J': total_energy,
            }
            final_msg = {'model_delta': model_delta, 'kpi_report': kpi_report}
            
            if self.send_msg(self.sock, final_msg, 'MSG_FINAL_CLIENT_REPORT_TO_EDGE'):
                self.logger.info("FedAvg model delta and KPIs successfully uploaded.")
            else:
                self.logger.error("Failed to upload FedAvg model delta.")

    def _execute_local_fedavg_training(self, local_iterations: int, optimizer: optim.Optimizer) -> Tuple[bool, Dict[str, Any]]:
        self.model.train()
        comp_start_time = time.perf_counter()
        
        # [优化] 在开始耗时的训练循环前，先打印一条日志
        self.logger.info(f"Starting local training for {local_iterations} batches...")
        
        try:
            # 使用itertools.cycle无限循环数据加载器，以应对迭代次数超过数据集大小的情况
            data_iterator = itertools.cycle(self.current_dataloader)
            for i in range(local_iterations):
                inputs, labels = next(data_iterator)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()

            computation_time = time.perf_counter() - comp_start_time

        except Exception as e:
            self.logger.error(f"Error during FedAvg local training: {e}", exc_info=True)
            return False, {}
            
        return True, {
            'computation_time_s': computation_time,
            'communication_time_s': 0.01, # 估算值
            'total_interaction_time_s': computation_time + 0.01
        }
    
    def _simulate_energy_fedavg(self, computation_time_s: float) -> Tuple[float, float, float]:
        # 模拟FedAvg模式下的能耗
        model_size_bytes = sum(p.numel() for p in self.model.parameters()) * BYTES_PER_FLOAT32
        channel_capacity_bps = (self.bandwidth * 1e6) * np.log2(1 + self.channel_quality)
        time_to_transmit = model_size_bytes * 8 / (channel_capacity_bps if channel_capacity_bps > 0 else 1e9)
        communication_energy = time_to_transmit * (RX_POWER_WATTS + TX_POWER_WATTS)
        computation_energy = computation_time_s * COMP_POWER_COEFF * (self.cpu_frequency ** 2)
        total_energy = communication_energy + computation_energy
        return computation_energy, communication_energy, total_energy

    def _send_status(self):
        # 更新并发送客户端的当前状态
        status_payload = self._update_and_get_client_status()
        msg_to_send = {'client_id': self.id, 'status_payload': status_payload}
        return self.send_msg(self.sock, msg_to_send, 'MSG_CLIENT_STATUS_TO_EDGE')


    def _setup_for_sfl_training(self, directive: dict):
        # 为一次SFL交互准备模型和优化器
        # 在创建新模型前，显式地删除旧的模型对象引用
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'anchor_model') and self.anchor_model is not None:
            del self.anchor_model
        
        self.optimizer = None 
        torch.cuda.empty_cache()

        received_model_state = directive.get('client_model_part_state_dict')
        self.local_iterations = directive.get('local_iterations')
        self.split_point = directive.get('model_split_point')
        anchor_model_state = directive.get('anchor_model_for_pullback')
        # 如果 directive 中包含算力/带宽分配，则采用该分配值覆盖本地的 cpu_frequency / bandwidth
        comp_alloc_directive = directive.get('comp_alloc')
        bw_alloc_directive = directive.get('bw_alloc')
        if comp_alloc_directive is not None:
            try:
                self.cpu_frequency = float(comp_alloc_directive)
                self.last_allocated_comp = self.cpu_frequency # [MHR-SFL] 记录分配值
            except Exception:
                self.logger.warning(f"Invalid comp_alloc in directive: {comp_alloc_directive}")
        if bw_alloc_directive is not None:
            try:
                self.bandwidth = int(bw_alloc_directive)
                self.last_allocated_bw = self.bandwidth # [MHR-SFL] 记录分配值
            except Exception:
                self.logger.warning(f"Invalid bw_alloc in directive: {bw_alloc_directive}")
        
        macro_policy = directive.get('macro_policy_package', {})
        self.mu = macro_policy.get('mu', 0.0)
        learning_rate = directive.get('sfl_lr', self.args.sfl_lr)
        model_name = self.args.model_name
        
        if any(v is None for v in [received_model_state, self.local_iterations, self.split_point]):
            self.logger.error("Invalid SFL training directive: missing key fields.")
            return False
        
        try:
            full_model_template = get_unified_model_constructor(model_name, self.args.input_channels, self.args.output_channels)
            self.model = get_unified_split_model_function(full_model_template, model_name, self.split_point)[0]
            self.model.load_state_dict(received_model_state)
            self.model.to(self.device)

            if anchor_model_state and self.mu > 0:
                self.anchor_model = get_unified_split_model_function(full_model_template, model_name, self.split_point)[0]
                self.anchor_model.load_state_dict(anchor_model_state)
                self.anchor_model.to(self.device).eval()
            else:
                self.anchor_model = None
        except Exception as e:
            self.logger.error(f"Failed to build/split model '{model_name}': {e}", exc_info=True)
            return False

        # 检查是否进入了新的全局回合，如果是，则清空旧的优化器状态
        # Check if it's a new global round; if so, reset the optimizer state
        if self.current_global_round != self.round_of_last_optimizer_state:
            self.logger.info(f"New global round ({self.current_global_round + 1}). Resetting optimizer state.")
            self.optimizer_state_for_round = None

        # 为新模型创建一个全新的优化器实例
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=5e-4)

        # 如果存在本回合的“记忆”（即非首次交互），则加载它
        if self.optimizer_state_for_round is not None:
            try:
                self.optimizer.load_state_dict(self.optimizer_state_for_round)
                self.logger.debug("Successfully loaded optimizer state from previous interaction in this round.")
            except ValueError:
                 self.logger.warning(
                    f"Could not load optimizer state due to a model structure mismatch "
                    f"(likely a new split point). Starting fresh for this interaction."
                )
                 self.optimizer_state_for_round = None
            except Exception as e:
                self.logger.warning(f"Could not load previous optimizer state, starting fresh. Reason: {e}")
                self.optimizer_state_for_round = None

        return True

    def _execute_local_sfl_training(self) -> Tuple[bool, Dict[str, Any]]:
        self.model.train()
        
        comp_time, comm_time, smashed_data_size, grad_size = 0.0, 0.0, 0, 0
        total_interaction_start = time.perf_counter()

        self.logger.info(f"Starting local training for {self.local_iterations} batches...")

        try:
            data_iterator = itertools.cycle(self.current_dataloader)
            for i in range(self.local_iterations): 
                inputs, labels = next(data_iterator)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()

                # --- 前向传播 ---
                t1 = time.perf_counter()
                smashed_data = self.model(inputs)
                comp_time += (time.perf_counter() - t1)

                # --- 与边缘服务器通信（发送激活值，接收梯度） ---
                t2 = time.perf_counter()
                smashed_data_cpu_clone = smashed_data.detach().clone().cpu()
                if smashed_data.requires_grad:
                    smashed_data.retain_grad()
                
                smashed_data_msg = {'outputs': smashed_data_cpu_clone, 'label': labels.cpu()}
                smashed_data_size += smashed_data.numel() * BYTES_PER_FLOAT32
                if not self.send_msg(self.sock, smashed_data_msg, 'MSG_SMASHED_DATA_TO_EDGE'): 
                    raise ConnectionError("Failed to send smashed data.")
                del smashed_data_cpu_clone, smashed_data_msg, inputs, labels
                
                msg_type, grad_msg = self.recv_msg(self.sock, timeout=300.0)
                if not grad_msg or msg_type != 'MSG_GRADIENT_TO_CLIENT':
                    raise ConnectionError(f"Did not receive valid gradient. Got '{msg_type}'.")
                comm_time += (time.perf_counter() - t2)
                
                # --- 反向传播 ---
                t3 = time.perf_counter()
                server_grad = grad_msg['gradient'].to(self.device)
                
                # 1. 首先，基于从服务器接收的主损失梯度进行反向传播
                smashed_data.backward(server_grad)
                
                # # 2. 接着，如果存在锚点模型和mu>0，计算并反向传播近端项的梯度。
                # #    这个梯度会自动累加到模型参数上。
                # if self.anchor_model and self.mu > 0:
                #     proximal_term = sum((local_param - anchor_param).pow(2).sum() for local_param, anchor_param in zip(self.model.parameters(), self.anchor_model.parameters()))
                #     proximal_loss = (self.mu / 2) * proximal_term
                #     proximal_loss.backward()
                
                comp_time += (time.perf_counter() - t3)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 将梯度的总范数限制在1.0以内
                self.optimizer.step()
                del server_grad, grad_msg, smashed_data
            
            # 在本地训练成功完成后，保存优化器的状态，并记录当前的全局回合数
            self.optimizer_state_for_round = self.optimizer.state_dict()
            self.round_of_last_optimizer_state = self.current_global_round

            self.logger.info(f"Finished {self.local_iterations} SFL batches. Total: {time.perf_counter() - total_interaction_start:.2f}s (Compute: {comp_time:.2f}s, Comm: {comm_time:.2f}s)")

        except ConnectionError as e: 
            self.logger.error(f"Connection error during local training: {e}"); return False, {}
        except Exception as e:
            self.logger.error(f"Unexpected error in local training: {e}", exc_info=True); return False, {}

        total_interaction_time = time.perf_counter() - total_interaction_start
        
        return True, {
            "computation_time_s": comp_time, 
            "communication_time_s": comm_time, 
            "total_interaction_time_s": total_interaction_time, 
            "smashed_data_size": smashed_data_size, 
            "grad_size": grad_size
        }

    def _upload_final_sfl_report(self, metrics: dict):
        # 上传SFL交互的最终结果
        if not self.model: return False

        comp_energy, comm_energy, total_energy = self._simulate_energy_sfl(metrics)
        kpi_report = {
            'utility': self.U,
            'total_data_size': len(self.full_local_data_indices),
            'computation_time_s': metrics['computation_time_s'],
            'communication_time_s': metrics['communication_time_s'],
            'total_interaction_time_s': metrics['total_interaction_time_s'],
            'computation_energy_J': comp_energy,
            'communication_energy_J': comm_energy,
            'total_energy_consumption_J': total_energy,
        }
        final_msg = {
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'kpi_report': kpi_report
        }
        
        if not self.send_msg(self.sock, final_msg, 'MSG_FINAL_CLIENT_REPORT_TO_EDGE'):
            self.logger.error("Failed to upload final SFL model and KPIs.")
            return False

        self.logger.debug("Final SFL report sent. Waiting for acknowledgement...")
        msg_type, _ = self.recv_msg(self.sock, timeout=30.0)
        if msg_type != 'MSG_EDGE_ACK_TO_CLIENT':
            self.logger.error(f"Did not receive correct acknowledgement. Expected 'MSG_EDGE_ACK_TO_CLIENT', got '{msg_type}'.")
            return False
        
        self.logger.info("Acknowledgement received. Standing by.")
        return True

    def _simulate_energy_sfl(self, metrics: dict) -> Tuple[float, float, float]:
        # 模拟SFL模式下的能耗
        final_model_size_bytes = sum(p.numel() for p in self.model.parameters()) * BYTES_PER_FLOAT32
        channel_capacity_bps = (self.bandwidth * 1e6) * np.log2(1 + self.channel_quality)
        
        time_up_smashed = metrics['smashed_data_size'] * 8 / (channel_capacity_bps if channel_capacity_bps > 0 else 1e9)
        time_down_grads = metrics['grad_size'] * 8 / (channel_capacity_bps if channel_capacity_bps > 0 else 1e9)
        time_up_model = final_model_size_bytes * 8 / (channel_capacity_bps if channel_capacity_bps > 0 else 1e9)

        communication_energy = (time_up_smashed + time_up_model) * TX_POWER_WATTS + (time_down_grads) * RX_POWER_WATTS
        computation_energy = metrics['computation_time_s'] * COMP_POWER_COEFF * (self.cpu_frequency ** 2)
        total_energy = communication_energy + computation_energy
        return computation_energy, communication_energy, total_energy
    
    def _update_and_get_client_status(self) -> Dict[str, Any]:
        # 更新并返回客户端的动态状态
        if self.quality_tier == 'low':
            self.cpu_frequency = round(random.uniform(0.5, 1.5), 2)
            self.channel_quality = round(random.uniform(50, 150), 2)
            self.bandwidth = random.randint(5, 10)
        elif self.quality_tier == 'medium':
            self.cpu_frequency = round(random.uniform(1.5, 2.5), 2)
            self.channel_quality = round(random.uniform(150, 300), 2)
            self.bandwidth = random.randint(10, 20)
        else:
            self.cpu_frequency = round(random.uniform(2.5, 3.5), 2)
            self.channel_quality = round(random.uniform(300, 500), 2)
            self.bandwidth = random.randint(20, 30)
            
        # [MHR-SFL] 如果存在上一轮的分配记录，则使用分配值覆盖随机生成的物理值，用于计算Utility
        # 这意味着Utility反映的是"如果继续使用上次分配的资源，预期的效用"
        if self.last_allocated_comp is not None:
            self.cpu_frequency = self.last_allocated_comp
        if self.last_allocated_bw is not None:
            self.bandwidth = self.last_allocated_bw
            
        self.distance = round(random.uniform(50, MAX_DISTANCE), 1)
        self.evaluate_training_utility()
        
        return {
            'client_id': self.id, 
            'utility': self.U, 
            'bandwidth': self.bandwidth,
            'comp_power': self.cpu_frequency, 
            'battery': self.battery_level,
            'data_size': len(self.full_local_data_indices), 
            'link_rate': self.channel_quality,
            'distance': self.distance, 
            'last_round_participated': self.last_round_participated,
            'quality_tier': self.quality_tier
        }

    def compute_class_distribution(self) -> Dict[int, int]:
        # 计算本地数据集的类别分布
        if not self.full_local_data_indices or not self.full_train_dataset_ref: return {}
        label_counts = Counter()
        try:
            if hasattr(self.full_train_dataset_ref, 'targets'):
                all_targets = np.array(self.full_train_dataset_ref.targets)
                subset_targets = all_targets[self.full_local_data_indices]
                label_counts.update(subset_targets.tolist())
            else: 
                for idx in self.full_local_data_indices:
                    _, label = self.full_train_dataset_ref[idx]
                    label_counts.update([label.item() if torch.is_tensor(label) else label])
        except Exception as e:
            self.logger.error(f"Error computing class distribution: {e}")
            return {}
        return dict(label_counts)

    def compute_entropy(self, class_distribution: Dict[int, int]) -> float:
        # 计算类别分布的熵
        if not class_distribution: return 0.0
        total = sum(class_distribution.values())
        if total == 0: return 0.0
        probs = np.array([v / total for v in class_distribution.values() if v > 0])
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

    def evaluate_training_utility(self):
        # 评估客户端的训练效用
        class_dist = self.compute_class_distribution()
        entropy = self.compute_entropy(class_dist)
        diversity_score = 1 + entropy
        data_size_score = len(self.full_local_data_indices) / MAX_DATASET_SIZE
        U_data = diversity_score * data_size_score
        num_classes = self.args.output_channels
        U_data = U_data / (1.0 + np.log2(num_classes))
        
        compute_score = (self.cpu_frequency / MAX_CPU_FREQUENCY) ** GAMMA_CPU
        comm_score = (self.channel_quality / MAX_CHANNEL_QUALITY) ** ETA_CHANNEL
        distance_penalty = 1 / (1 + LAMBDA_DIST * self.distance)
        U_sys = compute_score * comm_score * distance_penalty
        
        age_of_update = self.current_global_round - self.last_round_participated
        U_AoU = 1 / (1 + max(0, age_of_update))
        
        w_data = getattr(self.args, 'w_data', 0.4)
        w_sys = getattr(self.args, 'w_sys', 0.4)
        w_aou = getattr(self.args, 'w_aou', 0.2)
        
        self.U = w_data * U_data + w_sys * U_sys + w_aou * U_AoU

    def close_connection(self):
        # 安全地关闭网络连接
        if self.sock:
            self.logger.debug("Closing connection to edge server.")
            try:
                self.sock.close()
            except Exception:
                pass
            finally:
                self.sock = None