# --- 1. 全局默认配置 ---
# 这些是默认值，可以在执行时通过命令行参数覆盖
DATASET="cifar100"
MODEL_NAME="resnet18"
MODEL_DIR=".//home/lsy/UEC-SFL/mappo/non-iid/30"
NUM_CLIENTS=20
NUM_EDGES=4
NUM_AGENTS_PER_EDGE=5
SEED=4
TOTAL_GLOBAL_ROUNDS=60
LOG_BASE_DIR="./evaluation_results"

# --- 2. 默认数据分布配置 ---
# 默认使用 non-iid alpha=0.5
IID_FLAG="--alpha 0.5"
DIR_NAME="non_iid_alpha_0.5"
POLICY_ARG="" # 用于存储从命令行传入的单个策略

# --- 3. 【核心】智能参数解析 ---
# 这个循环会读取所有命令行参数, 并覆盖上面的默认值
# 例如: bash run.sh --iid 1 --num_clients 10 --test_policy ARES
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --iid)
            IID_FLAG="--iid $2"
            DIR_NAME="iid"
            shift; shift ;;
        --alpha)
            IID_FLAG="--alpha $2"
            DIR_NAME="non_iid_alpha_$2"
            shift; shift ;;
        --test_policy)
            POLICY_ARG="$2"
            shift; shift ;;
        *) # 其他所有无法识别的参数 (如 --num_clients 20) 都被收集起来
            OTHER_ARGS+=("$1")
            shift ;;
    esac
done


# --- 进程清理函数 ---
cleanup() {
    echo "" && echo ">>> Detected exit signal, cleaning up..." && pkill -P $$ && echo ">>> Cleanup complete." && exit
}
trap cleanup INT QUIT TERM EXIT

# --- 定义所有需要对比的算法策略 ---
policies_to_test=('FedAvg_Baseline' 'SplitFed' 'ARES' 'RAF-SFL' 'MADRL-SFL' 'HAC-SFL')
policies_to_test+=('MHR-SFL')

# --- 端口管理 ---
CLOUD_PORT_BASE=9100
EDGE_PORT_BASE=6000
PORT_OFFSET=0

CURRENT_CLOUD_PORT=$((CLOUD_PORT_BASE + PORT_OFFSET))
CURRENT_EDGE_BASE_PORT=$((EDGE_PORT_BASE + PORT_OFFSET * 100))

# =================================================================
#                  --- 实验主逻辑 ---
# =================================================================

if [ -n "$POLICY_ARG" ]; then
    # --- 路径A: 执行单个测试方案 ---
    policy=$POLICY_ARG
    echo "========================================================"
    echo "==> 执行单项测试方案: ${policy}"
    echo "==> 数据分布模式: ${DIR_NAME} (由参数 ${IID_FLAG} 控制)"
    echo "========================================================"

    LOG_DIR="${LOG_BASE_DIR}/${DIR_NAME}/${policy}"
    mkdir -p "$LOG_DIR"
    echo "--> 日志将保存至: ${LOG_DIR}"
    
    CMD="python -u main.py --experiment_name=${policy} --log_dir=${LOG_DIR} \
        --model_name=${MODEL_NAME} --dataset=${DATASET} --output_channels=100 \
        --cloud_port=${CURRENT_CLOUD_PORT} --server_port_base=${CURRENT_EDGE_BASE_PORT} \
        ${IID_FLAG} --test_policy ${policy} ${OTHER_ARGS[*]}"
        
    if [[ "$policy" == "HAC-SFL" || "$policy" == "MADRL-SFL" || "$policy" == "MHR-SFL" ]]; then
        if ! [[ " ${OTHER_ARGS[*]} " =~ " --model_dir " ]]; then
            CMD="$CMD --model_dir=${MODEL_DIR}"
        fi
    fi

    echo "Executing command: $CMD"
    eval $CMD

else
    # --- 路径B: 循环执行所有预设方案 ---
    echo "========================================================"
    echo "==> 未指定单项测试，将开始执行全量对比测试..."
    echo "==> 数据分布模式: ${DIR_NAME} (由参数 ${IID_FLAG} 控制)"
    echo "========================================================"
    
    for policy in "${policies_to_test[@]}"; do
        LOG_DIR="${LOG_BASE_DIR}/${DIR_NAME}/${policy}"
        mkdir -p "$LOG_DIR"
        echo "--------------------------------------------------------"
        echo "--> 正在测试方案: ${policy}"
        echo "--> 日志将保存至: ${LOG_DIR}"
        echo "--------------------------------------------------------"

        CMD="python -u main.py --experiment_name=${policy} --log_dir=${LOG_DIR} \
            --cloud_port=${CURRENT_CLOUD_PORT} --server_port_base=${CURRENT_EDGE_BASE_PORT} \
            --model_name=${MODEL_NAME} --dataset=${DATASET} --output_channels=100 \
            ${IID_FLAG} --test_policy ${policy} ${OTHER_ARGS[*]}"
            
        if [[ "$policy" == "HAC-SFL" || "$policy" == "MADRL-SFL" || "$policy" == "MHR-SFL" ]]; then
            if ! [[ " ${OTHER_ARGS[*]} " =~ " --model_dir " ]]; then
                CMD="$CMD --model_dir=${MODEL_DIR}"
            fi
        fi

        echo "Executing command: $CMD"
        $CMD &
        PORT_OFFSET=$((PORT_OFFSET + 1))
    done
    wait
    echo "========================================================"
    echo "==> 所有方案在 [${DIR_NAME}] 数据模式下的测试均已执行完毕。"
    echo "========================================================"
fi