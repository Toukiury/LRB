#!/bin/bash

# 数据集与训练比例配置
declare -A datasets

datasets["astar"]="0.001"
datasets["bzip"]="1"
datasets["gcc"]="1"
datasets["libquantum"]="1"
datasets["mcf"]="1"
# 可以添加更多数据集
# datasets["xalanc"]="1"

memory_window=1000000
model_fraction="1"
output_dir="stat/unified_lrb_test"
# quiet="--quiet"  # 移除静默模式
verbose="--verbose"  # 添加详细输出模式

# 加速参数
boost="--boost"  # 启用boost加速
num_workers="--num_workers 4"  # 使用4个工作线程
boost_fr="--boost_fr"  # 启用Follower&Robust算法的Belady trace加速

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/unified_lrb_test"
mkdir -p "$log_dir"
mkdir -p "$project_root/$output_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!datasets[@]}"; do
    fractions=(${datasets[$dataset]})
    for fraction in "${fractions[@]}"; do
        # 统一测试LRB和GuardLRB
        unified_log_file="$log_dir/${dataset}_${fraction}_unified.log"
        echo "统一测试LRB和GuardLRB: $dataset, 训练比例: $fraction, 日志: $unified_log_file"
        
        unified_cmd="--dataset $dataset --lrbcomplete --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir/unified --include_guard_lrb $boost $num_workers $boost_fr $verbose --relax_threshold 0.2 --enable_random_relax --phase_reset_percentage 0.7 --guard_weight 0.7"
        
        echo "执行命令: python -m benchmark.__main__ $unified_cmd"
        python -m benchmark.__main__ $unified_cmd | tee "$unified_log_file"
    done
done

echo "统一测试LRB和GuardLRB完成。"

# 生成结果汇总
echo "生成结果汇总..."
python -c "
import os
import pandas as pd
import glob

# 结果处理逻辑
result_files = glob.glob('$output_dir/unified/*/*.csv')

results = []

for result_file in result_files:
    dataset = os.path.basename(os.path.dirname(result_file))
    
    # 读取结果
    df = pd.read_csv(result_file)
    
    # 提取LRB和GuardLRB的命中率
    lrb_row = df[df['Name'] == 'LRB']
    guardlrb_row = df[df['Name'] == 'GuardLRB']
    
    if not lrb_row.empty and not guardlrb_row.empty:
        lrb_hitrate = lrb_row['Hit Rate'].iloc[0]
        guardlrb_hitrate = guardlrb_row['Hit Rate'].iloc[0]
        
        # 计算改进百分比
        improvement = ((guardlrb_hitrate - lrb_hitrate) / lrb_hitrate) * 100 if lrb_hitrate > 0 else 0
        
        results.append({
            'Dataset': dataset,
            'LRB Hit Rate': f'{lrb_hitrate:.4f}',
            'GuardLRB Hit Rate': f'{guardlrb_hitrate:.4f}',
            'Improvement': f'{improvement:.2f}%'
        })

# 创建汇总表格
if results:
    summary_df = pd.DataFrame(results)
    summary_file = '$output_dir/unified_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'结果汇总已保存到 {summary_file}')
    print(summary_df)
else:
    print('未找到结果文件进行比较')
" 