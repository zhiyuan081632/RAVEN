
# Convert mp4 file to ma4c/wav
./data/convert_mp4_to_wav.sh

# If you want to run all demo pipeline please using  "VoxCeleb2_train_1000.txt" in the config file


# Mix speech with noise for train/val data
python -W ignore utils/mix_speech_gpu.py

# Extract visual feature with different methods
## VSRiW
python data/extract_visual_features_VSRiW.py /mnt/e/data/VoxCeleb2/test/mp4 /mnt/e/data/VoxCeleb2/test/vsriw --num_workers 1

## TalkNet
python data/extract_visual_features_TalkNet.py /mnt/e/data/VoxCeleb2/test/mp4 /mnt/e/data/VoxCeleb2/test/TalkNet_feats --num_workers 1

## LoCoNet
python data/extract_visual_features_LocoNet.py /mnt/e/data/VoxCeleb2/test/mp4 /mnt/e/data/VoxCeleb2/test/LoCoNet_feats --num_workers 1

# AVHuBERT
python data/extract_visual_features_AVHuBERT.py /mnt/e/data/VoxCeleb2/test/mp4 /mnt/e/data/VoxCeleb2/test/AVHuBERT_feats --num_workers 1


# Generate test data 
## noise_only 
python -W ignore data/generate_test_data.py --condition noise_only --snr="mixed,-10,-5,0"

## one_interfering_speaker
python -W ignore data/generate_test_data.py --condition one_interfering_speaker --snr="mixed,-10,-5,0"

## three_interfering_speakers
python -W ignore data/generate_test_data.py --condition three_interfering_speakers --snr="mixed,-10,-5,0"
