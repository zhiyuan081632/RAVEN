"""
音视频语音增强推理脚本

输入MP4视频文件，提取视觉特征，对音频加噪后进行增强

用法:
    # 处理单个MP4文件 (默认SNR=0dB)
    python inference.py --input video.mp4 --output ./output --ckpt model.ckpt
    
    # 指定SNR和噪声文件
    python inference.py --input video.mp4 --output ./output --ckpt model.ckpt --snr -5 --noise noise.wav
    
    # 处理目录
    python inference.py --input ./videos --output ./enhanced --ckpt model.ckpt
    
    # 使用已有的视觉特征
    python inference.py --input video.mp4 --output ./output --ckpt model.ckpt --visual_feat features.npy
"""

import argparse
import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from models.fusion import Fusion
from utils.spec_audio_conversion import (
    convert_to_complex_spectrogram_with_compression_torch,
    convert_to_audio_from_complex_spectrogram_after_compression_torch
)


class AudioEnhancer:
    """音视频语音增强器"""
    
    def __init__(self, ckpt_path, visual_encoder, embedding_size=None, device="cuda"):
        self.device = device
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size or config.embedding_size_dict.get(visual_encoder, 512)
        self.sample_rate = 16000
        self.audio_duration = 5  # 5秒
        
        print(f"加载模型: {ckpt_path}")
        print(f"视觉编码器: {visual_encoder}, 嵌入维度: {self.embedding_size}")
        
        self.model = Fusion(embedding_size=self.embedding_size)
        
        # 加载checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 处理state_dict key前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        print("模型加载完成")
    
    def extract_audio_from_video(self, video_path):
        """从MP4中提取音频"""
        import subprocess
        import tempfile
        
        # 创建临时wav文件
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # 使用ffmpeg提取音频
        cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar {self.sample_rate} -ac 1 "{temp_wav.name}" -loglevel error'
        subprocess.run(cmd, shell=True, check=True)
        
        # 读取音频
        audio, sr = librosa.load(temp_wav.name, sr=self.sample_rate)
        os.unlink(temp_wav.name)  # 删除临时文件
        
        return audio
    
    def load_audio(self, audio_path):
        """加载音频文件"""
        if audio_path.endswith('.mp4'):
            audio = self.extract_audio_from_video(audio_path)
        else:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 裁剪或填充到5秒
        target_length = self.sample_rate * self.audio_duration
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        return audio
    
    def add_noise(self, clean_audio, noise_audio, snr_db):
        """
        给干净音频添加噪声
        
        Args:
            clean_audio: 干净音频
            noise_audio: 噪声音频
            snr_db: 信噪比 (dB)
        
        Returns:
            mixed_audio: 混合后的带噪音频
        """
        # 确保长度一致
        if len(noise_audio) < len(clean_audio):
            # 循环噪声
            repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
            noise_audio = np.tile(noise_audio, repeats)[:len(clean_audio)]
        else:
            noise_audio = noise_audio[:len(clean_audio)]
        
        # 计算能量
        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # 计算缩放因子
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(clean_power / (snr_linear * noise_power + 1e-8))
        
        # 混合
        mixed_audio = clean_audio + scale * noise_audio
        
        # 归一化防止溢出
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
        
        return mixed_audio
    
    def load_visual_features(self, video_path, visual_feat_path=None):
        """
        加载或提取视觉特征
        
        优先使用已有的npy文件，否则尝试从视频提取
        """
        feat_dir_map = {
            'VSRiW': 'vsriw',
            'TalkNet': 'TalkNet_feats',
            'LoCoNet': 'Loconet_feats',
            'AVHuBERT': 'AVHuBERT_feats'
        }
        feat_dir_name = feat_dir_map.get(self.visual_encoder, 'vsriw')
        
        # 1. 如果指定了特征文件路径
        if visual_feat_path and os.path.exists(visual_feat_path):
            print(f"  加载视觉特征: {visual_feat_path}")
            features = np.load(visual_feat_path)
        else:
            # 2. 尝试从标准路径推断
            feat_path = video_path.replace("/mp4/", f"/{feat_dir_name}/")
            feat_path = os.path.splitext(feat_path)[0] + ".npy"
            
            if os.path.exists(feat_path):
                print(f"  加载视觉特征: {feat_path}")
                features = np.load(feat_path)
            else:
                print(f"  警告: 视觉特征不存在, 使用零填充")
                print(f"  (如需提取特征，请先运行对应的特征提取脚本)")
                return torch.zeros(125, self.embedding_size, dtype=torch.float32)
        
        # 确保特征是125帧
        if len(features.shape) == 3:  # (1, T, D) -> (T, D)
            features = features.squeeze(0)
        
        if features.shape[0] > 125:
            features = features[:125]
        elif features.shape[0] < 125:
            pad = np.zeros((125 - features.shape[0], features.shape[-1]))
            features = np.vstack([features, pad])
        
        return torch.tensor(features, dtype=torch.float32)
    
    @torch.no_grad()
    def enhance(self, noisy_audio, visual_features):
        """增强音频"""
        audio = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        face_embed = visual_features.unsqueeze(0).to(self.device)
        
        # 转换为频谱
        mixed_mag_spec, mixed_phase = convert_to_complex_spectrogram_with_compression_torch(audio)
        
        # 调整维度: (B, N, T) -> (B, T, N)
        mixed_mag_spec = mixed_mag_spec.permute(0, 2, 1)
        mixed_phase = mixed_phase.permute(0, 2, 1)
        
        # 确保帧数是4的倍数
        padded_length = mixed_mag_spec.size(1) - mixed_mag_spec.size(1) % 4
        mixed_mag_spec = mixed_mag_spec[:, :padded_length, :]
        mixed_phase = mixed_phase[:, :padded_length, :]
        
        # 模型推理
        mag_spec_est = self.model(mixed_mag_spec, face_embed)
        
        # 转换回音频
        clean_complex_spec_est = (mag_spec_est, mixed_phase)
        enhanced_audio = convert_to_audio_from_complex_spectrogram_after_compression_torch(clean_complex_spec_est)
        
        return enhanced_audio.squeeze(0).cpu().numpy()
    
    def process_file(self, input_path, output_dir, noise_path=None, snr_db=0, visual_feat_path=None):
        """
        处理单个视频/音频文件
        
        Args:
            input_path: 输入MP4或音频文件
            output_dir: 输出目录
            noise_path: 噪声文件路径 (可选，不提供则使用白噪声)
            snr_db: 信噪比 (dB)
            visual_feat_path: 视觉特征文件路径 (可选)
        """
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 加载干净音频
            print(f"  提取音频...")
            clean_audio = self.load_audio(input_path)
            
            # 2. 加载噪声并混合
            print(f"  添加噪声 (SNR={snr_db}dB)...")
            if noise_path and os.path.exists(noise_path):
                noise, _ = librosa.load(noise_path, sr=self.sample_rate)
            else:
                # 使用白噪声
                noise = np.random.randn(len(clean_audio)) * 0.1
            
            noisy_audio = self.add_noise(clean_audio, noise, snr_db)
            
            # 3. 加载视觉特征
            print(f"  加载视觉特征...")
            visual_feat = self.load_visual_features(input_path, visual_feat_path)
            
            # 4. 增强
            print(f"  增强处理...")
            enhanced_audio = self.enhance(noisy_audio, visual_feat)
            
            # 5. 保存结果
            clean_path = os.path.join(output_dir, f"{base_name}_clean.wav")
            noisy_path = os.path.join(output_dir, f"{base_name}_noisy_snr{snr_db}.wav")
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced_snr{snr_db}.wav")
            
            sf.write(clean_path, clean_audio, self.sample_rate)
            sf.write(noisy_path, noisy_audio, self.sample_rate)
            sf.write(enhanced_path, enhanced_audio, self.sample_rate)
            
            print(f"  ✅ 保存:")
            print(f"     干净音频: {clean_path}")
            print(f"     带噪音频: {noisy_path}")
            print(f"     增强音频: {enhanced_path}")
            
            return True
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_directory(self, input_dir, output_dir, noise_path=None, snr_db=0):
        """处理目录中的所有视频文件"""
        video_files = glob(os.path.join(input_dir, '**', '*.mp4'), recursive=True)
        print(f"找到 {len(video_files)} 个视频文件")
        
        success_count = 0
        for video_path in tqdm(video_files, desc="处理中"):
            print(f"\n处理: {video_path}")
            rel_path = os.path.relpath(video_path, input_dir)
            sub_output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            
            if self.process_file(video_path, sub_output_dir, noise_path, snr_db):
                success_count += 1
        
        print(f"\n完成! 成功: {success_count}/{len(video_files)}")


def main():
    parser = argparse.ArgumentParser(description="音视频语音增强推理脚本")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入MP4视频文件或目录")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--ckpt", "-c", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--visual_encoder", "-v", type=str, default="VSRiW",
                        choices=["VSRiW", "TalkNet", "LoCoNet", "AVHuBERT"],
                        help="视觉编码器类型")
    parser.add_argument("--snr", type=int, default=0,
                        help="信噪比 (dB), 默认0")
    parser.add_argument("--noise", type=str, default=None,
                        help="噪声文件路径 (可选，不提供则使用白噪声)")
    parser.add_argument("--visual_feat", type=str, default=None,
                        help="视觉特征文件路径 (可选)")
    parser.add_argument("--embedding_size", type=int, default=None,
                        help="嵌入维度 (默认自动选择)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    # 创建增强器
    enhancer = AudioEnhancer(
        ckpt_path=args.ckpt,
        visual_encoder=args.visual_encoder,
        embedding_size=args.embedding_size,
        device=args.device
    )
    
    print(f"\n配置:")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  SNR: {args.snr} dB")
    print(f"  噪声: {args.noise or '白噪声'}")
    print()
    
    # 处理输入
    if os.path.isfile(args.input):
        print(f"处理文件: {args.input}")
        enhancer.process_file(args.input, args.output, args.noise, args.snr, args.visual_feat)
    elif os.path.isdir(args.input):
        print(f"处理目录: {args.input}")
        enhancer.process_directory(args.input, args.output, args.noise, args.snr)
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
