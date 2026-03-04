import torch
import torchaudio
import torchvision
import numpy as np

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)

class FixedVideoTransform:
    """修复版VideoTransform，正确处理张量维度"""
    def __init__(self, speed_rate):
        self.speed_rate = speed_rate
        
    def __call__(self, sample):
        # 输入应该是 (T, H, W, C) 或 (T, C, H, W) 格式
        if len(sample.shape) == 4:
            # 如果是 (T, H, W, C) 格式，转换为 (T, C, H, W)
            if sample.shape[-1] == 3:  # 最后一维是通道
                sample = sample.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # 现在sample应该是 (T, C, H, W) 格式
        # 应用速度调整
        if self.speed_rate != 1:
            indices = torch.linspace(0, sample.shape[0]-1, 
                                   int(sample.shape[0] / self.speed_rate), 
                                   dtype=torch.int64)
            sample = torch.index_select(sample, dim=0, index=indices)
        
        # 归一化和标准化
        sample = sample.float() / 255.
        sample = torchvision.transforms.CenterCrop(88)(sample)
        sample = torchvision.transforms.Normalize(0.421, 0.165)(sample)
        
        return sample

def extract_single_video_fixed(video_path, output_path=None):
    """使用修复版变换的单视频特征提取"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    sys.path.append(config.VSRIW_PATH)
    from pipelines.model import AVSR
    import torchvision
    
    print(f"=== 修复版单视频特征提取 ===")
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
        
        # 2. 简单预处理
        print("\n2. 简单预处理...")
        center_h, center_w = H // 2, W // 2
        crop_size = min(H, W) // 2
        start_h = max(0, center_h - crop_size // 2)
        start_w = max(0, center_w - crop_size // 2)
        end_h = min(H, start_h + crop_size)
        end_w = min(W, start_w + crop_size)
        
        cropped = video_tensor[:, start_h:end_h, start_w:end_w, :]
        print(f"   裁剪后形状: {cropped.shape}")
        
        # 缩放到标准尺寸
        import cv2
        resized_frames = []
        for frame in cropped:
            resized_frame = cv2.resize(frame, (88, 88))
            resized_frames.append(resized_frame)
        
        processed_video = np.array(resized_frames)
        print(f"   处理后形状: {processed_video.shape}")
        
        # 3. 使用修复版视频变换
        print("\n3. 视频变换(修复版)...")
        video_transform = FixedVideoTransform(speed_rate=1)
        transformed_video = video_transform(torch.tensor(processed_video))
        print(f"   变换后形状: {transformed_video.shape}")
        
        # 4. 确保正确的5维格式 (B, C, T, H, W)
        if len(transformed_video.shape) == 4:  # (T, C, H, W)
            transformed_video = transformed_video.permute(1, 0, 2, 3)  # (C, T, H, W)
            transformed_video = transformed_video.unsqueeze(0)  # (1, C, T, H, W)
            print(f"   调整为5维格式: {transformed_video.shape}")
        
        # 5. 特征提取
        print("\n4. 特征提取...")
        model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
        model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
        
        model = AVSR("video", model_path, model_conf, rnnlm=None, rnnlm_conf=None, 
                    penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device="cuda")
        
        with torch.no_grad():
            enc_feats = model.model.encode(transformed_video.to("cuda"), True)
        
        features = enc_feats.detach().cpu().numpy()
        print(f"   提取特征形状: {features.shape}")
        
        # 6. 保存npy文件
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
    test_video = "/mnt/e/data/VoxCeleb2/dev/mp4/id05750/iOKl9LefSvc/00300.mp4"
    print("=== 运行修复版测试 ===")
    success = extract_single_video_fixed(test_video)
    if success:
        print("\n🎉 修复版测试成功！")
    else:
        print("\n💥 修复版测试失败！")