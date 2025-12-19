#!/bin/bash

# =================================================================
#               --- 1. 全局配置 (请根据您的环境修改) ---
# =================================================================

# --- 实验维度定义 ---
# 您可以在这里添加或删除想要测试的项
# 数据集/客户端场景/策略配置
datasets_to_test=('cifar10')
client_scenarios_to_test=(10) # 客户端数量场景
policies_to_test=('FedAvg_Baseline' 'SplitFed' 'ARES' 'RAF-SFL')
# 'FedAvg_Baseline' 'SplitFed' 'ARES' 'RAF-SFL' 'MADRL-SFL' 'HAC-SFL'

# 定义数据分布场景，格式为 "目录名;传递给脚本的标志"
data_distributions=(
    "iid;--iid 1"
    "non-iid;--alpha 0.5"
)

# --- 路径配置 ---
# 【重要】预训练的 rmappo 模型存放的根目录
# 脚本将按此结构寻找模型: ${MAPPO_MODELS_BASE_DIR}/{分布目录名}/
MAPPO_MODELS_BASE_DIR="./mappo" 
# 实验结果 (logs, csv) 保存的根目录
LOG_ROOT_DIR="./results-1"

# --- 模型和训练配置 ---
MODEL_NAME="resnet18"
TOTAL_GLOBAL_ROUNDS=100
SEED=4

# --- 网络端口配置 (顺序执行，无需修改) ---
CLOUD_PORT=9001
EDGE_PORT_BASE=6100

# =================================================================
#               --- 2. 脚本核心逻辑  ---
# =================================================================

# --- 收集从命令行传入的、用于覆盖默认值的其他参数 ---
# 例如: bash run_all_experiments.sh --some_extra_arg value
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
    OTHER_ARGS+=("$1")
    shift
done

# --- 进程清理函数 ---
cleanup() {
    echo "" && echo ">>> 检测到退出信号，正在清理所有子进程..." && pkill -P $$ && echo ">>> 清理完毕。" && exit
}
trap cleanup INT QUIT TERM EXIT

echo "========================================================"
echo "==> 即将开始全自动化序贯实验..."
echo "==> 覆盖所有维度: 数据集, 客户端, 数据分布, 算法."
echo "========================================================"

# --- 最外层循环: 遍历数据集 ---
for dataset in "${datasets_to_test[@]}"; do
    
    # 根据数据集设置输出通道数
    if [[ "$dataset" == "cifar10" ]]; then
        output_channels=10
    elif [[ "$dataset" == "cifar100" ]]; then
        output_channels=100
    else
        echo "错误: 未知数据集 ${dataset}，跳过。"
        continue
    fi

    # --- 第二层循环: 遍历客户端场景 ---
    for clients in "${client_scenarios_to_test[@]}"; do
        
        # 根据客户端总数，配置边缘服务器和每个边缘服务器的客户端数
        # 您可以根据需求自定义这里的逻辑
        if [[ $clients -eq 1 ]]; then
            num_clients=20
            num_edges=4
            num_agents_per_edge=5
        elif [[ $clients -eq 10 ]]; then
            num_clients=40
            num_edges=4 # 假设10个客户端分给2个边缘
            num_agents_per_edge=5
        else
            echo "警告: 未为 ${clients} 个客户端定义分组策略，使用默认值。"
            num_clients=$clients
            num_edges=1
            num_agents_per_edge=$clients
        fi
        
        # --- 第三层循环: 遍历数据分布 ---
        for dist_setting in "${data_distributions[@]}"; do
            IFS=';' read -r DIR_NAME DIST_FLAG <<< "$dist_setting"

            # --- 第四层循环: 遍历所有算法 ---
            for policy in "${policies_to_test[@]}"; do
                
                echo ""
                echo "=============================================================================="
                echo ">>> 正在运行实验:"
                echo ">>>   - 数据集:     ${dataset}"
                echo ">>>   - 客户端场景: ${clients} (clients=${num_clients}, edges=${num_edges})"
                echo ">>>   - 数据分布:   ${DIR_NAME}"
                echo ">>>   - 算法策略:   ${policy}"
                echo "=============================================================================="

                # 动态构建日志保存路径
                LOG_DIR="${LOG_ROOT_DIR}/${dataset}/clients_${clients}/${DIR_NAME}/${policy}"
                mkdir -p "$LOG_DIR"
                echo "--> 日志将保存至: ${LOG_DIR}"

                # 构建基础命令
                CMD="python -u main.py --experiment_name=${policy} --log_dir=${LOG_DIR} \
                    --model_name=${MODEL_NAME} --dataset=${dataset} --output_channels=${output_channels} \
                    --num_clients=${num_clients} --num_edges=${num_edges} --epochs=${TOTAL_GLOBAL_ROUNDS} \
                    --cloud_port=${CLOUD_PORT} --server_port_base=${EDGE_PORT_BASE} \
                    ${DIST_FLAG} --test_policy ${policy} ${OTHER_ARGS[*]}"
                
                # [核心逻辑] 如果是需要加载模型的算法，动态构建模型路径
                if [[ "$policy" == "HAC-SFL" || "$policy" == "MADRL-SFL" || "$policy" == "MHR-SFL" ]]; then
                    # 根据配置规则，自动推断模型应该存放的路径
                    if [ "$num_clients" -eq 20 ]; then
                        if [ "$DIR_NAME" = "non-iid" ]; then
                            DIR_NAME="non-iid/40"
                        else
                            DIR_NAME="iid"
                        fi
                    elif [ "$num_clients" -eq 40 ]; then
                        if [ "$DIR_NAME" = "non-iid" ]; then
                            DIR_NAME="non-iid-10/40"
                        else
                            DIR_NAME="iid-10/40"
                        fi
                    else 
                        echo "警告: 未为 ${num_clients} 个客户端定义分组策略，使用默认值。"
                    fi

                    CURRENT_MODEL_DIR="${MAPPO_MODELS_BASE_DIR}/${DIR_NAME}/"
                    
                    if [ -d "$CURRENT_MODEL_DIR" ]; then
                        echo "--> 检测到 [${policy}] 策略, 将从以下目录加载模型: ${CURRENT_MODEL_DIR}"
                        CMD="$CMD --algorithm_name=rmappo --use_recurrent_policy --model_dir=${CURRENT_MODEL_DIR}"
                    else
                        echo "!!!!!!!!!!!!!! 警告 !!!!!!!!!!!!!!"
                        echo "--> 策略 [${policy}] 需要预训练模型，但路径不存在: ${CURRENT_MODEL_DIR}"
                        echo "--> 将跳过此实验..."
                        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        sleep 3
                        continue # 跳过当前算法，继续下一个
                    fi
                fi

                echo "--> 执行命令:"
                echo "$CMD"
                
                # 执行命令
                eval $CMD
                
                echo "--> 实验 [${policy} on ${dataset}, clients ${clients}, ${DIR_NAME}] 已完成，休眠5秒..."
                sleep 5

            done
        done
    done
done

echo ""
echo "********************************************************"
echo "*** 所有实验均已成功运行！             ***"
echo "********************************************************"