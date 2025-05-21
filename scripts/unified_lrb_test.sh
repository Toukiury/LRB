#!/bin/bash

# 数据集与训练比例配置
declare -A datasets



datasets["bzip"]="1"

memory_window=1000000
output_dir="stat/lrb_simple_test"
verbose="--verbose"  # 添加详细输出模式

# SimpleGuardLRB参数
simple_relax_times=5  # 可以设置为5或其他值
simple_relax_prob=0.0

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/lrb_simple_test"
mkdir -p "$log_dir"
mkdir -p "$project_root/$output_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!datasets[@]}"; do
    fractions=(${datasets[$dataset]})
    for fraction in "${fractions[@]}"; do
        # 测试LRB和SimpleGuardLRB
        log_file="$log_dir/${dataset}_${fraction}_simple.log"
        echo "测试LRB和SimpleGuardLRB: $dataset, 训练比例: $fraction, 日志: $log_file"
        
        # 注意: 只使用include_simple_guard_lrb参数，不包括include_guard_lrb参数
        cmd="--dataset $dataset --lrbcomplete --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir --include_simple_guard_lrb --simple_relax_times $simple_relax_times --simple_relax_prob $simple_relax_prob $verbose"
        
        echo "执行命令: python -m benchmark.__main__ $cmd"
        python -m benchmark.__main__ $cmd | tee "$log_file"
    done
done

echo "LRB和SimpleGuardLRB测试完成。"

# 生成结果汇总
echo "生成结果汇总..."
python -c "
import os
import pandas as pd
import glob
import re

# 结果处理逻辑
result_files = glob.glob('$output_dir/*/*.csv')

results = []

for result_file in result_files:
    dataset = os.path.basename(os.path.dirname(result_file))
    fraction = os.path.basename(result_file).replace('.csv', '').split('_')[-1]
    
    # 读取结果
    df = pd.read_csv(result_file)
    
    # 提取LRB和SimpleGuardLRB的命中率
    lrb_row = df[df['Name'] == 'LRB']
    
    # 匹配新的SimpleGuardLRB-RT{relax_times}格式
    simple_guardlrb_pattern = re.compile(r'SimpleGuardLRB-RT\d+')
    simple_guardlrb_row = df[df['Name'].str.contains('SimpleGuardLRB', regex=True)]
    
    if not lrb_row.empty and not simple_guardlrb_row.empty:
        lrb_hitrate = lrb_row['Hit Rate'].iloc[0]
        simple_guardlrb_hitrate = simple_guardlrb_row['Hit Rate'].iloc[0]
        simple_guardlrb_name = simple_guardlrb_row['Name'].iloc[0]
        
        # 提取relax_times值
        relax_times = $simple_relax_times
        if 'RT' in simple_guardlrb_name:
            relax_times = int(re.search(r'RT(\d+)', simple_guardlrb_name).group(1))
        
        # 计算改进百分比
        improvement = ((simple_guardlrb_hitrate - lrb_hitrate) / lrb_hitrate) * 100 if lrb_hitrate > 0 else 0
        
        results.append({
            'Dataset': dataset,
            'Fraction': fraction,
            'Relax Times': relax_times,
            'LRB Hit Rate': f'{lrb_hitrate:.4f}',
            'SimpleGuardLRB Hit Rate': f'{simple_guardlrb_hitrate:.4f}',
            'Improvement': f'{improvement:.2f}%'
        })

# 创建汇总表格
if results:
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(['Dataset', 'Fraction'])
    summary_file = '$output_dir/lrb_simple_comparison_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'结果汇总已保存到 {summary_file}')
    print(summary_df)
else:
    print('未找到结果文件进行比较')
"