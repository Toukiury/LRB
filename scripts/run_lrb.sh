#!/bin/bash

# 数据集与训练比例配置（可根据实际情况修改）
declare -A lrb_dict



lrb_dict["xalanc"]="0.01 0.07 0.1 1"



memory_window=1000000
model_fraction="1"
output_dir="stat/lrb"
quiet=""  # 不使用静默模式，保留输出

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main-guard"
log_dir="$project_root/logs/benchmark/lrb"  # 更改日志目录以区分
mkdir -p "$log_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!lrb_dict[@]}"; do
    fractions=(${lrb_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        log_file="$log_dir/${dataset}_${fraction}.log"  # 更改日志文件名
        echo "评测数据集: $dataset, 训练比例: $fraction, 日志: $log_file"
        
        # 构建参数，移除boost相关参数
        cmd_args="--dataset $dataset --real --pred lrb --memory_window $memory_window --model_fraction $fraction --dump_file --output_root_dir $output_dir $quiet"
        
        # 运行并使用tee同时输出到终端和日志文件
        echo "执行命令: python -m benchmark.__main__ $cmd_args | tee $log_file"
        python -m benchmark.__main__ $cmd_args | tee "$log_file"
    done
done

echo "LRB基准测试（无boost加速）全部完成。"
echo "日志保存在 $log_dir 目录。" 