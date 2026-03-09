# Test audio-visual speech enhancement
# 支持三种测试场景: noise_only, one_interfering_speaker, three_interfering_speakers
visual_encoder="VSRiW_TalkNet_concatenate"
ckpt_path="/mnt/e/project/prjANS/src/AVSE/train/ckpt_20260304134032/VSRiW_TalkNet_concatenate_5layer/last.ckpt"
target_file="samples/007_26_M_YS_001_FACE.mp4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
output_dir="./output/${TIMESTAMP}"
mkdir -p "${output_dir}"

# ============================================================
# 1. noise_only: 目标语音 + 环境噪声
# ============================================================
echo "=== Test: noise_only ==="
python inference.py \
    --visual_encoder "$visual_encoder" \
    --ckpt_path "$ckpt_path" \
    --input "$target_file" \
    --condition noise_only \
    --noise samples/noise-free-sound-0000.wav \
    --snr -5 \
    --output ${output_dir}/noise_only

# ============================================================
# 2. one_interfering_speaker: 目标语音 + 1个干扰说话人
# ============================================================
echo "=== Test: one_interfering_speaker ==="
python inference.py \
    --visual_encoder "$visual_encoder" \
    --ckpt_path "$ckpt_path" \
    --input "$target_file" \
    --condition one_interfering_speaker \
    --interfering samples/019_24_F_LY_002_FACE.mp4 \
    --snr -5 \
    --output ${output_dir}/one_interfering_speaker

# ============================================================
# 3. three_interfering_speakers: 目标语音 + 3个干扰说话人
# ============================================================
echo "=== Test: three_interfering_speakers ==="
python inference.py \
    --visual_encoder "$visual_encoder" \
    --ckpt_path "$ckpt_path" \
    --input "$target_file" \
    --condition three_interfering_speakers \
    --interfering samples/019_24_F_LY_002_FACE.mp4 samples/00095.mp4 samples/00261.mp4 \
    --snr -5 \
    --output ${output_dir}/three_interfering_speakers
