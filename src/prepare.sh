
# Convert full data from mp4 to ma4c/wav
./data/extract_vox2_mp4_to_m4a.sh

# Run Demo pipeline with small data if needed ?
##  Split small data from full data 
python data/split_small_data.py
cp data/split_small.csv data/split.csv
cp data/split_small.parquet data/split.parquet

# Mix speech with noise in train/val data
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
