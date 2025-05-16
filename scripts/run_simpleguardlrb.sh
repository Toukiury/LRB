#!/bin/bash

# 默认参数
DATASET="xalanc"
MEMORY_WINDOW=1000000
RELAX_TIMES=5
RELAX_PROB=0.0
FOLLOW_IF_GUARDED=false
DISABLE_ADMISSION=false
DUMP_FILE=false
DEBUG=false
QUIET=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --memory-window)
      MEMORY_WINDOW="$2"
      shift 2
      ;;
    --relax-times)
      RELAX_TIMES="$2"
      shift 2
      ;;
    --relax-prob)
      RELAX_PROB="$2"
      shift 2
      ;;
    --follow-if-guarded)
      FOLLOW_IF_GUARDED=true
      shift
      ;;
    --disable-admission)
      DISABLE_ADMISSION=true
      shift
      ;;
    --dump-file)
      DUMP_FILE=true
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --quiet)
      QUIET=true
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 构建命令
CMD="python -m benchmark --dataset $DATASET --simpleguardlrb --memory_window $MEMORY_WINDOW --relax_times $RELAX_TIMES --relax_prob $RELAX_PROB"

# 添加可选参数
if [ "$FOLLOW_IF_GUARDED" = true ]; then
  CMD="$CMD --follow_if_guarded"
fi

if [ "$DISABLE_ADMISSION" = true ]; then
  CMD="$CMD --disable_admission"
fi

if [ "$DUMP_FILE" = true ]; then
  CMD="$CMD --dump_file"
fi

if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug"
fi

if [ "$QUIET" = true ]; then
  CMD="$CMD --quiet"
fi

# 打印并执行命令
echo "执行命令: $CMD"
eval $CMD 