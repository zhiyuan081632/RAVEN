import os
import torch
import numpy as np
import sys
import cv2

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sys.path.append(config.VSRIW_PATH)
from pipelines.model import AVSR
from pipelines.data.transforms import VideoTransform
import torchvision

def compare_videos_analysis(video1_path, video2_path):
    """对比分析两个视频的处理差异"""
    print("=== 视频对比分析 ===\n")
    
    videos = [
        ("问题视频", video1_path),
        ("参考视频", video2_path)
    ]
    
    results = {}
    
    for name, video_path in videos:
        print(f"--- {name} ---")
        print(f"路径: {video_path}")
        
        if not os.path.exists(video_path):
            print("❌ 文件不存在\n")
            continue
            
        try:
            # 1. 基本信息
            size = os.path.getsize(video_path)
            print(f"文件大小: {size} bytes ({size/1024/1024:.1f} MB)")
            
            # 2. 视频读取
            video_tensor = torchvision.io.read_video(video_path, pts_unit='sec')[0].numpy()
            T, H, W, C = video_tensor.shape
            print(f"视频形状: ({T}, {H}, {W}, {C})")
            print(f"时长: {T/25:.1f} 秒")
            
            # 3. 简单预处理
            center_h, center_w = H // 2, W // 2
            crop_size = min(H, W) // 2
            start_h = max(0, center_h - crop_size // 2)
            start_w = max(0, center_w - crop_size // 2)
            end_h = min(H, start_h + crop_size)
            end_w = min(W, start_w + crop_size)
            
            cropped = video_tensor[:, start_h:end_h, start_w:end_w, :]
            print(f"裁剪后: {cropped.shape}")
            
            # 缩放
            resized_frames = []
            for frame in cropped:
                resized_frame = cv2.resize(frame, (88, 88))
                resized_frames.append(resized_frame)
            
            processed_video = np.array(resized_frames)
            print(f"处理后: {processed_video.shape}")
            
            # 4. 视频变换
            processed_tensor = torch.tensor(processed_video).permute(0, 3, 1, 2)
            print(f"张量格式: {processed_tensor.shape}")
            
            video_transform = VideoTransform(speed_rate=1)
            try:
                transformed_video = video_transform(processed_tensor)
                print(f"✅ 变换成功: {transformed_video.shape}")
                
                # 5. 模型输入格式检查
                if len(transformed_video.shape) == 4:  # (T, C, H, W)
                    final_input = transformed_video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
                    print(f"最终输入格式: {final_input.shape}")
                    
                    # 6. 模型测试（小批次）
                    model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
                    model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
                    
                    model = AVSR("video", model_path, model_conf, rnnlm=None, rnnlm_conf=None, 
                                penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device="cuda")
                    
                    # 只测试前几帧以节省时间
                    test_input = final_input[:, :, :min(50, final_input.shape[2]), :, :]  # 最多50帧
                    print(f"测试输入: {test_input.shape}")
                    
                    with torch.no_grad():
                        enc_feats = model.model.encode(test_input.to("cuda"), True)
                        features = enc_feats.detach().cpu().numpy()
                        print(f"✅ 特征提取成功: {features.shape}")
                        results[name] = "SUCCESS"
                        
                else:
                    print(f"❌ 变换后形状异常: {transformed_video.shape}")
                    results[name] = "SHAPE_ERROR"
                    
            except Exception as e:
                print(f"❌ 变换失败: {str(e)[:100]}...")
                results[name] = f"TRANSFORM_ERROR: {str(e)[:50]}"
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)[:100]}...")
            results[name] = f"PROCESS_ERROR: {str(e)[:50]}"
        
        print()
    
    # 总结
    print("=== 对比结果 ===")
    for name, result in results.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"{status} {name}: {result}")

if __name__ == "__main__":
    problem_video = "/mnt/e/data/VoxCeleb2/dev/mp4/id05750/iOKl9LefSvc/00300.mp4"
    reference_video = "/mnt/e/data/VoxCeleb2/dev/mp4/id04219/bGLFntUuA58/00408.mp4"
    
    compare_videos_analysis(problem_video, reference_video)