
# Convert mp4 file to ma4c/wav
# ./data/convert_mp4_to_wav.sh

# 配置开关: "small" 或 "full"
MODE="small"

if [ "$MODE" = "small" ]; then
    echo "使用小量数据模式"
    ##  Split small data from full data 
    python data/extract_small_data.py
    cp data/split_small.csv data/split.csv
elif [ "$MODE" = "full" ]; then
    echo "使用完整数据模式"
    cp data/split_full.csv data/split.csv
fi


# Mix speech with noise for train/val data
python -W ignore utils/mix_speech_gpu.py

# Extract visual feature with different methods
## VSRiW
python data/VSRiW_extract_visual_features.py --speech_dataset VoxCeleb2 --split train

## TalkNet
python data/TalkNet_extract_visual_features.py --speech_dataset VoxCeleb2 --split train

## LoCoNet
python data/LoCoNet_extract_visual_features.py --speech_dataset VoxCeleb2 --split train

# AVHuBERT
python data/AVHuBERT_extract_visual_features.py --speech_dataset VoxCeleb2 --split train


# Generate test data 
## noise_only 
python -W ignore data/generate_test_data.py --condition noise_only --snr="mixed,-10,-5,0"

## one_interfering_speaker
python -W ignore data/generate_test_data.py --condition one_interfering_speaker --snr="mixed,-10,-5,0"

## three_interfering_speakers
python -W ignore data/generate_test_data.py --condition three_interfering_speakers --snr="mixed,-10,-5,0"
