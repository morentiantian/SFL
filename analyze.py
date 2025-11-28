import pandas as pd
import os
import numpy as np
from itertools import product # 引入 product 来方便地生成列顺序

# --- 1. 配置 ---

# 基础目录, 脚本将在此文件夹内查找数据
base_dir = 'data_picture'

# 定义实验的所有维度
scenario_folders = [
    'cifar10',
    'cifar10-10',
    'cifar100',
    'cifar100-10'
]
distributions = ['iid', 'non_iid']

# 【最终表格行顺序】按此顺序排列算法
algorithms_in_order = [
    'FedAvg_Baseline',
    'SplitFed',
    'ARES',
    'RAF-SFL',
    'MADRL-SFL',
    'HAC-SFL'
]

# 定义三个数据源的文件名
file_names = {
    'acc': 'accuracy_loss_summary_table.csv',
    'time': 'time_summary_table.csv',
    'energy': 'energy_summary_table.csv'
}

# --- 2. 核心数据提取逻辑 ---

# 使用一个字典来存储所有提取到的扁平化数据
# 格式: {(场景, 分布, 算法, 指标): 值}
results_data = {}

print("--- 开始汇总所有实验结果 (Acc, Time, Energy) ---")

# 遍历所有实验组合
for scenario in scenario_folders:
    for dist in distributions:
        path_prefix = os.path.join(base_dir, scenario, dist)
        print(f"\n正在处理目录: {path_prefix}")

        # --- a. 读取准确率数据 (数值不变) ---
        try:
            acc_df = pd.read_csv(os.path.join(path_prefix, file_names['acc']), header=[0, 1], index_col=0)
            for algo in algorithms_in_order:
                # 检查列是否存在
                if ('test_accuracy', algo) in acc_df.columns:
                    max_acc = acc_df[('test_accuracy', algo)].max()
                    results_data[(scenario, dist, algo, 'Acc')] = max_acc
                else:
                    results_data[(scenario, dist, algo, 'Acc')] = np.nan
        except FileNotFoundError:
            print(f"  [警告] 准确率文件未找到。")
            for algo in algorithms_in_order: results_data[(scenario, dist, algo, 'Acc')] = np.nan
        except Exception as e: 
            print(f"  [错误] 读取准确率文件时出错: {e}")

        # --- b. 读取时间数据 (数值除以100) ---
        try:
            time_df = pd.read_csv(os.path.join(path_prefix, file_names['time']), index_col='algorithm')
            for algo in algorithms_in_order:
                interaction_time = time_df.loc[algo, '总交互时间 (s)']
                # 【修改点】将时间值除以100
                results_data[(scenario, dist, algo, 'Times')] = interaction_time / 100
        except FileNotFoundError:
            print(f"  [警告] 时间文件未找到。")
            for algo in algorithms_in_order: results_data[(scenario, dist, algo, 'Times')] = np.nan
        except Exception as e: 
            print(f"  [错误] 读取时间文件时出错: {e}")

        # --- c. 读取能耗数据 (数值除以100) ---
        try:
            energy_df = pd.read_csv(os.path.join(path_prefix, file_names['energy']), index_col='algorithm')
            for algo in algorithms_in_order:
                total_energy = energy_df.loc[algo, '总能耗 (J)']
                # 【修改点】将能耗值除以100
                results_data[(scenario, dist, algo, 'Energy')] = total_energy / 100
        except FileNotFoundError:
            print(f"  [警告] 能耗文件未找到。")
            for algo in algorithms_in_order: results_data[(scenario, dist, algo, 'Energy')] = np.nan
        except Exception as e: 
            print(f"  [错误] 读取能耗文件时出错: {e}")

# --- 3. 将数据重构为最终表格 ---

print("\n--- 所有数据提取完毕，正在生成最终汇总报告 ---")

if not results_data:
    print("错误：未能提取到任何数据，无法生成报告。")
else:
    # 将字典转换为带有多级索引的Pandas Series
    s = pd.Series(results_data)
    s.index.names = ['scenario', 'distribution', 'algorithm', 'metric']
    
    # 使用 unstack 将索引层级转换为列层级
    final_df = s.unstack(level=['scenario', 'distribution', 'metric'])

    # --- 严格按照您期望的顺序定义和排序所有列 ---
    
    # 1. 定义场景文件夹到(客户端, 数据集)的映射关系
    def map_scenario(scenario_str):
        if scenario_str == 'cifar10': return ('客户端5', 'cifar10')
        elif scenario_str == 'cifar10-10': return ('客户端10', 'cifar10')
        elif scenario_str == 'cifar100': return ('客户端5', 'cifar100')
        elif scenario_str == 'cifar100-10': return ('客户端10', 'cifar100')
        return ('未知', '未知')

    # 2. 应用映射，重构列索引
    new_cols_tuples = []
    for scenario, dist, metric in final_df.columns:
        client, dataset = map_scenario(scenario)
        new_cols_tuples.append((client, dataset, dist, metric))
        
    final_df.columns = pd.MultiIndex.from_tuples(
        new_cols_tuples,
        names=['场景', 'dataset', 'distribution', 'metric']
    )
    
    # 3. 定义每一级表头的期望顺序
    client_order = ['客户端5', '客户端10']
    dataset_order = ['cifar10', 'cifar100']
    dist_order = ['iid', 'non_iid']
    metric_order = ['Acc', 'Times', 'Energy'] # <-- 指标顺序

    # 4. 生成所有可能的、符合期望顺序的列名
    final_column_order = list(product(client_order, dataset_order, dist_order, metric_order))
    
    # 5. 过滤掉数据中不存在的列，并按最终顺序重排列
    existing_columns_in_order = [col for col in final_column_order if col in final_df.columns]
    final_df = final_df[existing_columns_in_order]
    
    # 6. 按最终顺序重排DataFrame的行
    final_df = final_df.reindex(algorithms_in_order) 

    print("最终报告预览:")
    print(final_df.to_string())

    # --- 4. 保存到Excel或CSV文件 ---
    
    # 方案A：保存为Excel (需要 openpyxl，效果最好)
    output_csv_file = 'final_summary_report.csv'
    try:
        # 我们使用 to_csv() 来替代 to_excel()
        final_df.to_csv(output_csv_file)
        print(f"\n✅ 成功！最终汇总报告已保存到: {output_csv_file}")
        print("提示: CSV格式不支持复杂表头，但在Excel等软件中打开时通常可以正确解析。")
    except Exception as e:
        print(f"\n[错误] 保存CSV文件失败: {e}")