import multiprocessing
import os
import random
import sys
import time
import numpy as np
import torch
import logging
import copy
from pathlib import Path
from multiprocessing import Barrier

sys.path.append(os.getcwd())

from config import args_parser
from utils.logger_setup import setup_logger, suppress_stdout 
from datasets.sampling import get_dataset_cached, get_static_datasets_indices, prepare_validation_loader
from value.client import Client
from value.cloud import CloudServer
from value.edge import EdgeServer

# 云服务器进程的执行目标函数
# Execution target function for the cloud server process
def cloud_process_target(args):
    log_file = os.path.join(args.log_dir, 'cloud.log')
    setup_logger('CloudProcess', log_file)
    logging.captureWarnings(True)
    logger = logging.getLogger('CloudProcess')
    try:
        logger.info("Cloud process starting...")
        cloud = CloudServer(host='127.0.0.1', port=args.cloud_port, args=args)
        cloud.accept_edges_initial()
        cloud.run(total_global_rounds=args.epochs)
        logger.info("Cloud process finished its run successfully.")
    except Exception as e:
        logger.critical(f"Unhandled exception in cloud process: {e}", exc_info=True)
    finally:
        logger.info("Cloud process terminating.")

# 单个客户端进程的执行目标函数
# Execution target function for a single client process
def client_process_target(args, client_id, client_subset_indices, barrier):
    client_args = copy.deepcopy(args)
    client_args.client_id = client_id
    logger_name = f"Client_{client_id}"

    log_file = os.path.join(args.log_dir, f'client_{client_id}.log')
    setup_logger(logger_name, log_file)
    logging.captureWarnings(True)
    logger = logging.getLogger(logger_name)

    try:
        client = Client(client_args, local_data_indices=client_subset_indices)
        
        logger.info(f"Client {client_id} is ready and waiting at the barrier.")
        barrier.wait()
        logger.info(f"Client {client_id} passed the barrier, starting main loop.")
        
        client.start()
    except Exception as e:
        logger.critical(f"Unhandled exception in client {client_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Client process {client_id} terminating.")

# 单个边缘节点 (智能服务器) 进程的执行目标函数
# Execution target function for a single edge node (intelligent server) process
def edge_server_process_target(args, edge_id, barrier):
    edge_args = copy.deepcopy(args)
    edge_args.edge_id = edge_id
    run_dir = Path(edge_args.log_dir)
    
    logger_name = f"EdgeServer-{edge_id}"
    log_file_path = run_dir / f"edge_server_{edge_id}.log"
    setup_logger(name=logger_name, log_file=log_file_path)
    server_logger = logging.getLogger(logger_name)

    try:
        server_logger.info(f"Intelligent Edge Server process {edge_id} starting...")
        
        torch.manual_seed(edge_args.seed + edge_id * 10)
        np.random.seed(edge_args.seed + edge_id * 10)
        random.seed(edge_args.seed + edge_id * 10)
        
        if edge_args.cuda and torch.cuda.is_available():
            gpu_id = edge_id % torch.cuda.device_count()
            edge_args.device = torch.device(f"cuda:{gpu_id}")
            server_logger.info(f"Assigning process to GPU {gpu_id}.")
        else:
            edge_args.device = torch.device("cpu")

        # 1. 直接创建智能化的 EdgeServer 实例
        # 1. Directly create an intelligent EdgeServer instance
        val_loader = prepare_validation_loader(args, edge_id, server_logger)
        edge_server = EdgeServer(args=edge_args, logger_ref=server_logger, val_loader=val_loader)

        server_logger.info(f"Edge Server {edge_id} is ready and waiting at the barrier.")
        barrier.wait()
        server_logger.info(f"Edge Server {edge_id} passed the barrier, starting test execution.")
        
        # ======================= [核心修正] =======================
        # 在开始主循环之前，必须先连接到云服务器
        # Before starting the main loop, it is necessary to connect to the cloud server
        if not edge_server.connect_to_cloud(cloud_host='127.0.0.1', cloud_port=args.cloud_port):
            server_logger.critical("Failed to establish connection with the cloud server. Terminating.")
            return # 无法连接则直接退出
        # ==========================================================

        # 2. 在测试开始前，一次性接受所有客户端的持久连接
        # 2. At the beginning of the test, accept persistent connections from all clients at once
        num_clients_per_edge = args.num_clients // args.num_edges
        edge_server.accept_persistent_connections(num_expected_clients=num_clients_per_edge)

        # 3. 主测试循环：直接驱动 EdgeServer 完成所有全局轮次
        # 3. Main test loop: directly drive the EdgeServer to complete all global rounds
        for global_round in range(args.epochs):
            server_logger.info(f"\n[Edge {edge_id}] === Starting Global Round [{global_round + 1}/{args.epochs}] ===")
            
            success, _ = edge_server.receive_global_config_from_cloud()
            if not success:
                server_logger.error("Failed to get config from cloud, terminating test.")
                break
            
            edge_server.evaluate_baseline_on_validation_set()
            
            edge_server.execute_all_sfl_interactions_for_global_round()

            if not edge_server.finalize_and_report_to_cloud():
                server_logger.error("Failed to report to cloud, terminating test.")
                break

            if not edge_server.wait_for_round_completion_signal():
                server_logger.error("Did not receive round completion signal, terminating test.")
                break
        
        server_logger.info(f"--- Edge {edge_id} completed all {args.epochs} test rounds ---")
        edge_server.shutdown()
        
    except Exception as e:
        server_logger.critical(f"Edge Server process {edge_id} encountered a fatal error: {e}", exc_info=True)
    finally:
        server_logger.info(f"Edge Server process {edge_id} is terminating.")


if __name__ == '__main__':
    if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    else:
        multiprocessing.set_start_method('forkserver', force=True)

    parser = args_parser()
    args = parser.parse_args()
    
    args.log_dir = Path(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main_logger = setup_logger('MainLauncher', os.path.join(args.log_dir, 'main_launch.log'))

    main_logger.info("======================================================")
    main_logger.info(f"Starting experiment: {args.experiment_name}")
    main_logger.info(f"Test Policy: {args.test_policy}")
    data_dist_info = f"IID" if args.iid == 1 else f"Non-IID (alpha={args.alpha})"
    main_logger.info(f"Data Distribution: {data_dist_info}")
    main_logger.info("======================================================")

    main_logger.info("Loading and partitioning dataset indices...")
    try:
        with suppress_stdout():
            trainset_full, _ = get_dataset_cached(name=args.dataset, root=args.data_dir)
        
        client_indices_list = get_static_datasets_indices(
            trainset_full, 
            args.num_clients, 
            iid=(args.iid == 1),
            alpha=args.alpha
        )
        client_subsets_dict = {i: indices for i, indices in enumerate(client_indices_list)}
        main_logger.info(f"Dataset indices partitioned for {len(client_subsets_dict)} clients.")
    except Exception as e:
        main_logger.critical(f"Failed to load or partition dataset: {e}", exc_info=True)
        sys.exit(1)

    num_parties = args.num_clients + args.num_edges
    barrier = Barrier(num_parties)
    main_logger.info(f"Synchronization barrier created for {num_parties} parties.")
    
    processes = []

    main_logger.info("Starting Cloud process...")
    cloud_proc = multiprocessing.Process(target=cloud_process_target, args=(args,))
    processes.append(cloud_proc)
    cloud_proc.start()
    time.sleep(3)

    main_logger.info(f"Starting {args.num_clients} Client processes...")
    for i in range(args.num_clients):
        if i in client_subsets_dict:
            client_proc = multiprocessing.Process(
                target=client_process_target, 
                args=(args, i, client_subsets_dict[i], barrier)
            )
            processes.append(client_proc)
            client_proc.start()

    main_logger.info(f"Starting {args.num_edges} Edge Server processes...")
    for i in range(args.num_edges):
        edge_proc = multiprocessing.Process(
            target=edge_server_process_target,
            args=(args, i, barrier)
        )
        processes.append(edge_proc)
        edge_proc.start()

    main_logger.info("All processes launched. Main process will now wait for Cloud to complete.")
    
    cloud_proc.join()
    main_logger.info("Cloud process has completed.")

    main_logger.info("Terminating all remaining client and edge processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
    
    main_logger.info("All processes have been terminated. Experiment run is complete.")