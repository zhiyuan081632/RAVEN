#!/bin/bash

# 测试条件
test_conditions=("noise_only" "one_interfering_speaker" "three_interfering_speakers")
test_snrs=("-10" "-5" "0" "mixed")

# 模型路径
visual_encoder="VSRiW_TalkNet_concatenate"
ckpt_path="/mnt/e/project/prjANS/src/AVSE/train/ckpt_20260304134032/VSRiW_TalkNet_concatenate_5layer/last.ckpt"

# 遍历组合
for condition in "${test_conditions[@]}"; do
    for snr in "${test_snrs[@]}"; do
        echo "Running test with condition: $condition, SNR: $snr"
        
        # 记录开始时间
        start_time=$(date +%s)
        
        python -W ignore test.py \
            --visual_encoder "$visual_encoder" \
            --test_condition "$condition" \
            --test_snr "$snr" \
            --ckpt_path "$ckpt_path"
        
        # 记录结束时间
        end_time=$(date +%s)
        
        # 计算耗时
        duration=$((end_time - start_time))
        
        if [ $? -eq 0 ]; then
            echo "Test completed successfully: condition=$condition, SNR=$snr, Duration: ${duration}s"
        else
            echo "Test failed: condition=$condition, SNR=$snr, Duration: ${duration}s"
        fi
        
        echo "--------------------------------------------------"
    done
done

echo "All tests completed."
