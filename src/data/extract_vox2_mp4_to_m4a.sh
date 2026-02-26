#!/bin/bash
# LRS2 Audio Extraction Script
# This script extracts audio from LRS2 video files

# Save original terminal settings
ORIG_STTY=$(stty -g 2>/dev/null)

# Function to restore terminal
restore_terminal() {
    echo ""
    # Restore terminal settings
    if [ -n "$ORIG_STTY" ]; then
        stty "$ORIG_STTY" 2>/dev/null
    fi
    # Reset terminal attributes
    tput sgr0 2>/dev/null || true
    stty sane 2>/dev/null || true
    # Re-enable echo
    stty echo 2>/dev/null || true
}

# Trap to ensure terminal is restored on exit
trap 'restore_terminal; exit' INT TERM EXIT

# Configuration
INPUT_DIR="/mnt/e/data/VoxCeleb2/dev/mp4"
OUTPUT_FORMAT="m4a"  # Options: wav, m4a
OUTPUT_DIR="/mnt/e/data/VoxCeleb2/dev/aac"
SAMPLE_RATE=16000
NUM_WORKERS=16

echo "=========================================="
echo "VoxCeleb2 Audio Extraction ($OUTPUT_FORMAT)"
echo "========================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Sample Rate: $SAMPLE_RATE Hz"
echo "Workers: $NUM_WORKERS"
echo "=========================================="
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Python script
python preprocess_vox2_audio.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --sample_rate $SAMPLE_RATE \
    --output_format $OUTPUT_FORMAT \
    --num_workers $NUM_WORKERS

# Capture exit code
EXIT_CODE=$?

# Always restore terminal state after script execution
restore_terminal

# Check for errors
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Script failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "Done! Audio files saved to: $OUTPUT_DIR"
