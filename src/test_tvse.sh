# Test audio-visual speech enhancement
# 支持三种测试场景: noise_only, one_interfering_speaker, three_interfering_speakers

CKPT="/mnt/e/project/prjANS/src/AVSE/train/ckpt_20260304134032/VSRiW_TalkNet_concatenate_5layer/last.ckpt"
TARGET="samples/007_26_M_YS_001_FACE.mp4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./output/${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}"

# ============================================================
# 1. noise_only: 目标语音 + 环境噪声
# ============================================================
echo "=== Test: noise_only ==="
python inference.py \
    --visual_encoder VSRiW_TalkNet_concatenate \
    --ckpt_path "$CKPT" \
    --input "$TARGET" \
    --condition noise_only \
    --noise samples/noise-free-sound-0000.wav \
    --snr -5 \
    --output ${OUTPUT_BASE}/noise_only

# ============================================================
# 2. one_interfering_speaker: 目标语音 + 1个干扰说话人
# ============================================================
echo "=== Test: one_interfering_speaker ==="
python inference.py \
    --visual_encoder VSRiW_TalkNet_concatenate \
    --ckpt_path "$CKPT" \
    --input "$TARGET" \
    --condition one_interfering_speaker \
    --interfering samples/019_24_F_LY_002_FACE.mp4 \
    --snr -5 \
    --output ${OUTPUT_BASE}/one_speaker

# ============================================================
# 3. three_interfering_speakers: 目标语音 + 3个干扰说话人
# ============================================================
echo "=== Test: three_interfering_speakers ==="
python inference.py \
    --visual_encoder VSRiW_TalkNet_concatenate \
    --ckpt_path "$CKPT" \
    --input "$TARGET" \
    --condition three_interfering_speakers \
    --interfering samples/019_24_F_LY_002_FACE.mp4 samples/00095.mp4 samples/00261.mp4 \
    --snr -5 \
    --output ${OUTPUT_BASE}/three_speakers
