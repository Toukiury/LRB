#!/bin/bash

# 激活conda环境
echo "激活conda环境Alice..."
eval "$(conda shell.bash hook)"
conda activate Alice

# GuardLRB算法基准测试脚本
declare -A guardlrb_dict

# 配置数据集和训练比例
guardlrb_dict["bzip"]="1"
# guardlrb_dict["libq"]="1" 

# Guard特有参数
memory_window=1000000
phase_reset_percentage=0.5
guard_weight=0.8
relax_threshold=0.3
output_dir="stat"
quiet=""

# 项目根目录
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/guardlrb"
mkdir -p "$log_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!guardlrb_dict[@]}"; do
    fractions=(${guardlrb_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        log_file="$log_dir/${dataset}_${fraction}.log"
        echo "评测数据集: $dataset, 训练比例: $fraction, 日志: $log_file"
        
        # 构建参数 - 使用--real和--pred lrb参数，确保Guard[LRB]变体显示在结果中
        cmd_args="--boost --boost_fr --dataset $dataset --real --pred lrb --memory_window $memory_window --model_fraction $fraction --phase_reset_percentage $phase_reset_percentage --guard_weight $guard_weight --relax_threshold $relax_threshold --dump_file --output_root_dir $output_dir $quiet"
        
        # 运行
        echo "执行命令: python -m benchmark.__main__ $cmd_args"
        python -m benchmark.__main__ $cmd_args | tee "$log_file"
    done
done

echo "GuardLRB基准测试全部完成。" 