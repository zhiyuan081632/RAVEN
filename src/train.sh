#!/bin/bash

# 配置开关: "small" 或 "full"
MODE="small"

if [ "$MODE" = "small" ]; then
    echo "使用小量数据模式"
    cp data/split_small.parquet data/split.parquet
    cp data/split_small.csv data/split.csv
elif [ "$MODE" = "full" ]; then
    echo "使用完整数据模式"
    cp data/split_full.parquet data/split.parquet
    cp data/split_full.csv data/split.csv
fi

python -W ignore train.py
