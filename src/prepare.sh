
# Convert full data from mp4 to ma4c/wav
./data/extract_vox2_mp4_to_m4a.sh

# 配置开关: "small" 或 "full"
MODE="small"

if [ "$MODE" = "small" ]; then
    echo "使用小量数据模式"
    ##  Split small data from full data 
    python data/extract_small_data.py
    cp data/split_small.parquet data/split.parquet
    cp data/split_small.csv data/split.csv
elif [ "$MODE" = "full" ]; then
    echo "使用完整数据模式"
    cp data/split_full.parquet data/split.parquet
    cp data/split_full.csv data/split.csv
fi


# Mix speech with noise for train/val data
python -W ignore utils/mix_speech_gpu.py

# Extract visual feature with different methods
# VSRiW
python data/VSRiW_extract_visual_features.py 

# TalkNet
python data/TalkNet_extract_visual_features.py

# LoCoNet
python data/LoCoNet_extract_visual_features.py

# AVHuBERT
python data/AVHuBERT_extract_visual_features.py


# Generate test data
python -W ignore data/generate_test_data.py --condition=noise_only --snr=-10
