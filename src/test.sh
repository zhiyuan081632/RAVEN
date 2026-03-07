python -W ignore test.py \
    --visual_encoder AVHuBERT_TalkNet_concatenate \
    --test_condition one_interfering_speaker \
    --test_snr -10 \
    --ckpt_path "/mnt/e/project/prjANS/src/AVSE/RAVEN/src/checkpoints/ckpt_20260307112935/AVHuBERT_TalkNet_concatenate_5layer/last.ckpt"
