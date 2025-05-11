#!/bin/bash

# 统一数据集和训练比例
declare -A dataset_dict

dataset_dict["bzip"]="1"


# 统一预测器列表
predictors=("gbm" "parrot" "lrb" "pleco" "popu" "pleco-bin" "oracle_bin" "oracle_dis")

# 其它通用参数
memory_window=1000000
output_dir="stat"
project_root="/home/disk2/tangwenzheng/cache-coliseum-main"
log_dir="$project_root/logs/benchmark/all_predictors"
mkdir -p "$log_dir"

cd "$project_root"
echo "当前工作目录: $(pwd)"

for dataset in "${!dataset_dict[@]}"; do
    fractions=(${dataset_dict[$dataset]})
    for fraction in "${fractions[@]}"; do
        for predictor in "${predictors[@]}"; do
            log_file="$log_dir/${dataset}_${fraction}_${predictor}.log"
            echo "评测数据集: $dataset, 训练比例: $fraction, 预测器: $predictor, 日志: $log_file"
            
            # 只允许--real或--oracle出现一个
            pred_mode="--real"
            extra_args=""
            if [[ "$predictor" == "oracle_bin" || "$predictor" == "oracle_dis" ]]; then
                pred_mode="--oracle"
                if [[ "$predictor" == "oracle_bin" ]]; then
                    extra_args="--noise_type bin"
                else
                    extra_args="--noise_type dis"
                fi
            fi
            if [[ "$predictor" == "lrb" ]]; then
                extra_args="$extra_args --memory_window $memory_window --lrb_only"
            fi
            if [[ "$predictor" == "gbm" ]]; then
                extra_args="$extra_args --memory_window $memory_window"
            fi

            cmd_args="--dataset $dataset $pred_mode --pred $predictor --model_fraction $fraction --dump_file --output_root_dir $output_dir $extra_args"
            
            # 运行
            echo "执行命令: python -m benchmark.__main__ $cmd_args"
            python -m benchmark.__main__ $cmd_args | tee "$log_file"
        done
    done
done

echo "所有预测器基准测试全部完成。" 