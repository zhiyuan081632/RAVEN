# Test audio and video speech enhancement
python inference.py \
    --visual_encoder VSRiW_TalkNet_concatenate \
    --ckpt_path "/mnt/e/project/prjANS/src/AVSE/train/ckpt_20260304134032/VSRiW_TalkNet_concatenate_5layer/last.ckpt" \
    --input samples/019_24_F_LY_002_FACE.mp4 \
    --noise samples/noise-free-sound-0001.wav \
    --snr -5 \
    --output ./output 
