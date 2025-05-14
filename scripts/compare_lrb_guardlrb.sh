#!/bin/bash

# 数据集与训练比例配置
declare -A datasets

datasets["bzip"]="1"
# 可以添加更多数据集
# datasets["xalanc"]="1"
# datasets["astar"]="1"

memory_window=1000000
model_fraction="1"
output_dir="stat/lrb_guardlrb_comparison"
quiet=""

# 加速参数
boost="--boost"  # 启用boost加速
num_workers="--num_workers 4"  # 使用4个工作线程
boost_fr="--boost_fr"  # 启用Follower&Robust算法的Belady trace加速

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/lrb_guardlrb_comparison"
mkdir -p "$log_dir"
mkdir -p "$project_root/$output_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!datasets[@]}"; do
    fractions=(${datasets[$dataset]})
    for fraction in "${fractions[@]}"; do
        # 原始LRB测试
        lrb_log_file="$log_dir/${dataset}_${fraction}_lrb.log"
        echo "评测LRB: $dataset, 训练比例: $fraction, 日志: $lrb_log_file"
        
        lrb_cmd="--dataset $dataset --real --pred lrb --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir/lrb $boost $num_workers $boost_fr --lrb_only $quiet"
        
        echo "执行命令: python -m benchmark.__main__ $lrb_cmd"
        python -m benchmark.__main__ $lrb_cmd | tee "$lrb_log_file"
        
        # GuardLRB测试
        guardlrb_log_file="$log_dir/${dataset}_${fraction}_guard_lrb.log"
        echo "评测GuardLRB: $dataset, 训练比例: $fraction, 日志: $guardlrb_log_file"
        
        guardlrb_cmd="--dataset $dataset --guard_lrb --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir/guardlrb $boost $num_workers $boost_fr $quiet"
        
        echo "执行命令: python -m benchmark.__main__ $guardlrb_cmd"
        python -m benchmark.__main__ $guardlrb_cmd | tee "$guardlrb_log_file"
        
        # 可以添加结果比较逻辑
        echo "比较 $dataset ($fraction) LRB vs GuardLRB 的结果..."
    done
done

echo "LRB和GuardLRB比较测试完成。"

# 可以添加结果汇总代码
echo "生成结果汇总..."
python -c "
import os
import pandas as pd
import glob

# 结果处理逻辑
lrb_files = glob.glob('$output_dir/lrb/*/*.csv')
guardlrb_files = glob.glob('$output_dir/guardlrb/*/*.csv')

results = []

for lrb_file in lrb_files:
    dataset = os.path.basename(os.path.dirname(lrb_file))
    algo_name = os.path.basename(lrb_file).replace('.csv', '')
    
    # 读取LRB结果
    lrb_df = pd.read_csv(lrb_file)
    lrb_hitrate = lrb_df['hit_rate'].iloc[0] if not lrb_df.empty else 0
    
    # 查找对应的GuardLRB结果
    guardlrb_file = lrb_file.replace('/lrb/', '/guardlrb/')
    if os.path.exists(guardlrb_file):
        guardlrb_df = pd.read_csv(guardlrb_file)
        guardlrb_hitrate = guardlrb_df['hit_rate'].iloc[0] if not guardlrb_df.empty else 0
        
        # 计算改进百分比
        improvement = ((guardlrb_hitrate - lrb_hitrate) / lrb_hitrate) * 100 if lrb_hitrate > 0 else 0
        
        results.append({
            'Dataset': dataset,
            'Algorithm': algo_name,
            'LRB Hit Rate': f'{lrb_hitrate:.4f}',
            'GuardLRB Hit Rate': f'{guardlrb_hitrate:.4f}',
            'Improvement': f'{improvement:.2f}%'
        })

# 创建汇总表格
if results:
    summary_df = pd.DataFrame(results)
    summary_file = '$output_dir/comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'结果汇总已保存到 {summary_file}')
    print(summary_df)
else:
    print('未找到结果文件进行比较')
" 