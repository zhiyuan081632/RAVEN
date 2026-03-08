python -W ignore test.py \
    --visual_encoder AVHuBERT_TalkNet_concatenate \
    --test_condition one_interfering_speaker \
    --test_snr -10 \
    --ckpt_path "/mnt/e/project/prjANS/src/AVSE/train/ckpt_20260308000953/AVHuBERT_VSRiW_concatenate_5layer/last.ckpt"
