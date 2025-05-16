#!/bin/bash

# 数据集与训练比例配置（可根据实际情况修改）
declare -A lrb_dict

lrb_dict["xalanc"]="0.01 0.07 0.1 1"

memory_window=1000000
model_fraction="1"
output_dir="stat/simpleguardlrb"
quiet="--quiet"

# SimpleGuardLRB特有参数
simple_relax_times=0
simple_relax_prob=0.0
follow_if_guarded=false
disable_admission=false

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/simpleguardlrb"
mkdir -p "$log_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!lrb_dict[@]}"; do
    fractions=(${lrb_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        log_file="$log_dir/${dataset}_${fraction}.log"
        echo "评测数据集: $dataset, 训练比例: $fraction, 日志: $log_file"
        
        # 构建参数 - 使用guard_lrb而不是simpleguardlrb
        cmd_args="--dataset $dataset --guard_lrb --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir $quiet"
        
        # 添加SimpleGuardLRB特有参数
        if [ "$simple_relax_times" -gt 0 ]; then
            cmd_args="$cmd_args --simple_relax_times $simple_relax_times"
        fi
        
        if (( $(echo "$simple_relax_prob > 0.0" | bc -l) )); then
            cmd_args="$cmd_args --simple_relax_prob $simple_relax_prob"
        fi
        
        if [ "$follow_if_guarded" = true ]; then
            cmd_args="$cmd_args --follow_if_guarded"
        fi
        
        if [ "$disable_admission" = true ]; then
            cmd_args="$cmd_args --disable_admission"
        fi
        
        # 运行评测
        echo "运行命令: python -m benchmark $cmd_args"
        python -m benchmark $cmd_args > "$log_file" 2>&1
        
        # 检查运行结果
        if [ $? -eq 0 ]; then
            echo "评测完成: $dataset, $fraction"
        else
            echo "评测失败: $dataset, $fraction, 请查看日志: $log_file"
        fi
    done
done

echo "所有评测完成，结果保存在: $output_dir" 