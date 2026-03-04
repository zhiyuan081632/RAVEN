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

def extract_single_video_fast(video_path, output_path=None):
    """快速版本：跳过复杂的人脸检测，直接使用简单处理"""
    print(f"=== 快速单视频特征提取 ===")
    print(f"输入视频: {video_path}")
    
    if not os.path.exists(video_path):
        print("❌ 视频文件不存在")
        return False
    
    try:
        # 1. 读取视频
        print("\n1. 读取视频...")
        video_tensor = torchvision.io.read_video(video_path, pts_unit='sec')[0].numpy()
        print(f"   视频形状: {video_tensor.shape}")
        
        T, H, W, C = video_tensor.shape
        duration = T / 25.0
        print(f"   视频时长: {duration:.2f}秒")
        print(f"   分辨率: {W}x{H}")
        
        # 2. 简单预处理（跳过复杂的人脸检测）
        print("\n2. 简单预处理...")
        # 直接中心裁剪和缩放
        center_h, center_w = H // 2, W // 2
        crop_size = min(H, W) // 2
        start_h = max(0, center_h - crop_size // 2)
        start_w = max(0, center_w - crop_size // 2)
        end_h = min(H, start_h + crop_size)
        end_w = min(W, start_w + crop_size)
        
        cropped = video_tensor[:, start_h:end_h, start_w:end_w, :]
        print(f"   裁剪后形状: {cropped.shape}")
        
        # 缩放到标准尺寸
        resized_frames = []
        for frame in cropped:
            resized_frame = cv2.resize(frame, (88, 88))
            resized_frames.append(resized_frame)
        
        processed_video = np.array(resized_frames)
        print(f"   处理后形状: {processed_video.shape}")
        
        # 3. 视频变换
        print("\n3. 视频变换...")
        # 将numpy数组转换为正确的torch格式 (T, H, W, C) -> (T, C, H, W)
        processed_tensor = torch.tensor(processed_video).permute(0, 3, 1, 2)  # (T, C, H, W)
        print(f"   转换后张量形状: {processed_tensor.shape}")
        
        video_transform = VideoTransform(speed_rate=1)
        transformed_video = video_transform(processed_tensor)
        print(f"   变换后形状: {transformed_video.shape}")
        
        # 确保正确的5维格式
        if len(transformed_video.shape) == 4:
            transformed_video = transformed_video.unsqueeze(1)
            print(f"   添加通道维度后: {transformed_video.shape}")
        
        # 4. 特征提取
        print("\n4. 特征提取...")
        model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
        model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
        
        model = AVSR("video", model_path, model_conf, rnnlm=None, rnnlm_conf=None, 
                    penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device="cuda")
        
        with torch.no_grad():
            enc_feats = model.model.encode(transformed_video.unsqueeze(0).to("cuda"), True)
        
        features = enc_feats.detach().cpu().numpy()
        print(f"   提取特征形状: {features.shape}")
        
        # 5. 保存npy文件
        if output_path is None:
            output_path = video_path.replace("/mp4/", "/vsriw/").replace(".mp4", ".npy")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, features)
        print(f"\n✅ 特征已保存到: {output_path}")
        print(f"   文件大小: {os.path.getsize(output_path)} bytes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试用例
    test_videos = [
        '/mnt/e/data/VoxCeleb2/dev/mp4/id05750/iOKl9LefSvc/00300.mp4',  # 问题视频
        '/mnt/e/data/VoxCeleb2/dev/mp4/id04219/bGLFntUuA58/00408.mp4'   # 参考视频
    ]

for video_path in test_videos:
    success = extract_single_video_fast(video_path)
    if success:
        print("\n🎉 快速测试成功！")
    else:
        print("\n💥 快速测试失败！")