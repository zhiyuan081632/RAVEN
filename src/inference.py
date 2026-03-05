"""
音视频语音增强推理脚本

输入MP4视频文件，提取视觉特征，对音频加噪后进行增强
支持从预生成npy加载特征，也支持从MP4在线提取

支持三种测试场景:
    - noise_only: 目标语音 + 环境噪声
    - one_interfering_speaker: 目标语音 + 1个干扰说话人
    - three_interfering_speakers: 目标语音 + 3个干扰说话人

用法:
    # 噪声场景 (默认)
    python inference.py --input video.mp4 --output ./output --ckpt_path model.ckpt --noise noise.wav --snr -5
    
    # 1个干扰说话人
    python inference.py --input video.mp4 --output ./output --ckpt_path model.ckpt \
        --condition one_interfering_speaker --interfering speaker.mp4 --snr -5
    
    # 3个干扰说话人
    python inference.py --input video.mp4 --output ./output --ckpt_path model.ckpt \
        --condition three_interfering_speakers --interfering s1.mp4 s2.mp4 s3.mp4 --snr -5
"""

import argparse
import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import torchvision
from glob import glob
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from models.fusion import Fusion
from utils.spec_audio_conversion import (
    convert_to_complex_spectrogram_with_compression_torch,
    convert_to_audio_from_complex_spectrogram_after_compression_torch
)
from torchmetrics.audio import (
    SignalNoiseRatio,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility
)


# ============================================================
# 在线视觉特征提取器
# 编码器	提取流程	输出维度
# VSRiW	  MP4 → 5s裁剪 → MediaPipe人脸检测 → VideoProcess → VideoTransform → AVSR.encode()	(T, 512)
# TalkNet MP4 → 5s裁剪 → 灰度化 → CenterCrop(112) → 归一化 → visualFrontend	(125, 512)
# ============================================================

def random_5s_clip(video, frame_rate=25):
    """裁剪/填充视频到5秒 (125帧)"""
    T = video.shape[0]
    target_frames = 5 * frame_rate
    if T >= target_frames:
        return video[:target_frames]
    else:
        pad_shape = list(video.shape)
        pad_shape[0] = target_frames - T
        pad_array = np.zeros(pad_shape, dtype=video.dtype)
        return np.concatenate((video, pad_array), axis=0)


class VSRiWExtractor:
    """VSRiW 视觉特征在线提取器"""
    
    def __init__(self, device="cuda"):
        from mediapipe.python.solutions.face_detection import FaceDetection, FaceKeyPoint
        sys.path.append(config.VSRIW_PATH)
        from pipelines.model import AVSR
        from pipelines.detectors.mediapipe.video_process import VideoProcess
        from pipelines.data.transforms import VideoTransform
        
        self.device = device
        self.FaceKeyPoint = FaceKeyPoint
        
        # 人脸检测器
        self.short_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.long_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=1)
        
        # 视频处理
        self.video_processor = VideoProcess()
        self.video_transform = VideoTransform(speed_rate=1)
        
        # VSRiW编码器模型
        model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
        model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
        self.model = AVSR("video", model_path, model_conf,
                          rnnlm=None, rnnlm_conf=None,
                          penalty=0.0, ctc_weight=0.1, lm_weight=0.0,
                          beam_size=40, device=device)
        print("  VSRiW 提取器已加载")
    
    def _detect(self, video_frames, detector):
        """检测人脸关键点"""
        FaceKeyPoint = self.FaceKeyPoint
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].y * ih)],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks
    
    @torch.no_grad()
    def extract(self, video_path):
        """
        从MP4在线提取VSRiW特征
        
        Returns:
            features: numpy array (1, T, 512) 或 None (失败时)
        """
        # 读取视频
        video = torchvision.io.read_video(video_path, pts_unit='sec')[0].numpy()
        video = random_5s_clip(video)
        
        # 人脸检测
        landmarks = self._detect(video, self.long_range_detector)
        if all(l is None for l in landmarks):
            landmarks = self._detect(video, self.short_range_detector)
            if all(l is None for l in landmarks):
                print(f"    VSRiW: 人脸检测失败")
                return None
        
        # 视频处理 + 变换
        video_tensor = self.video_processor(video, landmarks)
        if video_tensor is None:
            print(f"    VSRiW: 视频预处理失败")
            return None
        
        video_tensor = self.video_transform(torch.tensor(video_tensor))
        
        # 编码
        enc_feats = self.model.model.encode(video_tensor.to(self.device), True)
        return enc_feats.detach().cpu().numpy()  # (1, T, 512)


class TalkNetExtractor:
    """TalkNet 视觉特征在线提取器"""
    
    def __init__(self, device="cuda"):
        sys.path.append(config.TALKNET_PATH)
        from talkNet import talkNet
        
        self.device = device
        model_path = os.path.join(config.TALKNET_PATH, "pretrain_TalkSet.model")
        self.model = talkNet().to(device)
        self.model.loadParameters(model_path)
        self.model.eval()
        print("  TalkNet 提取器已加载")
    
    @torch.no_grad()
    def extract(self, video_path):
        """
        从MP4在线提取TalkNet特征
        
        Returns:
            features: numpy array (125, 512) 或 None (失败时)
        """
        from torchvision.transforms.functional import center_crop
        
        # 读取视频
        video = torchvision.io.read_video(video_path, pts_unit='sec')[0]  # (T, H, W, C)
        video = random_5s_clip(video.numpy())
        video = torch.tensor(video, dtype=torch.float32)
        
        # 转灰度
        rgb_weight = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 1, 1, 3)
        gray_video = torch.sum(video * rgb_weight, dim=-1, keepdim=False)  # (125, H, W)
        gray_video = center_crop(gray_video, [112, 112])
        
        # 归一化并通过模型
        gray_video = gray_video.unsqueeze(0)  # (1, 125, 112, 112)
        B, T, H, W = gray_video.shape
        gray_video = gray_video.to(self.device)
        gray_video = gray_video.view(B * T, 1, 1, W, H)
        gray_video = (gray_video / 255 - 0.4161) / 0.1688
        
        video_embed = self.model.model.visualFrontend(gray_video)
        video_embed = video_embed.view(B, T, -1)
        video_embed = video_embed.squeeze(0)  # (125, 512)
        
        return video_embed.detach().cpu().numpy()


class AudioEnhancer:
    """音视频语音增强器"""
    
    def __init__(self, ckpt_path, visual_encoder, embedding_size=None, 
                 device="cuda", extract_online=True):
        self.device = device
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size or config.embedding_size_dict.get(visual_encoder, 512)
        self.sample_rate = 16000
        self.audio_duration = 5  # 5秒
        self.extract_online = extract_online
        
        # 特征提取器 (懒加载)
        self._vsriw_extractor = None
        self._talknet_extractor = None
        
        print(f"加载Fusion模型: {ckpt_path}")
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
        print("Fusion模型加载完成")
        
        # 性能指标
        self.metrics = {
            'snr': SignalNoiseRatio().to(device),
            'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb').to(device),
            'sisdr': ScaleInvariantSignalDistortionRatio().to(device),
            'estoi': ShortTimeObjectiveIntelligibility(16000, True).to(device),
        }
    
    def _get_extractor(self, encoder_name):
        """懒加载特征提取器"""
        if encoder_name == 'VSRiW':
            if self._vsriw_extractor is None:
                print("  初始化 VSRiW 提取器...")
                self._vsriw_extractor = VSRiWExtractor(self.device)
            return self._vsriw_extractor
        elif encoder_name == 'TalkNet':
            if self._talknet_extractor is None:
                print("  初始化 TalkNet 提取器...")
                self._talknet_extractor = TalkNetExtractor(self.device)
            return self._talknet_extractor
        else:
            print(f"  警告: {encoder_name} 不支持在线提取")
            return None
    
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
    
    def mix_audio(self, clean_audio, condition, snr_db, noise_path=None, interfering_paths=None):
        """
        根据测试条件混合音频
        
        Args:
            clean_audio: 干净目标音频
            condition: 测试条件 (noise_only / one_interfering_speaker / three_interfering_speakers)
            snr_db: 信噪比 (dB)
            noise_path: 噪声文件路径 (noise_only时用)
            interfering_paths: 干扰说话人MP4/音频路径列表
        
        Returns:
            mixed_audio: 混合后的音频
            condition_desc: 条件描述字符串
        """
        if condition == "noise_only":
            # 噪声场景
            if noise_path and os.path.exists(noise_path):
                noise, _ = librosa.load(noise_path, sr=self.sample_rate)
            else:
                noise = np.random.randn(len(clean_audio)) * 0.1
            mixed = self.add_noise(clean_audio, noise, snr_db)
            desc = f"noise_only (noise={noise_path or 'white_noise'})"
            
        elif condition == "one_interfering_speaker":
            # 1个干扰说话人
            if not interfering_paths or len(interfering_paths) < 1:
                print("  警告: 未提供干扰说话人，使用白噪声")
                noise = np.random.randn(len(clean_audio)) * 0.1
                mixed = self.add_noise(clean_audio, noise, snr_db)
                desc = "one_interfering_speaker (fallback: white_noise)"
            else:
                interfering_audio = self.load_audio(interfering_paths[0])
                interfering_audio = librosa.util.normalize(interfering_audio)
                mixed = self.add_noise(clean_audio, interfering_audio, snr_db)
                desc = f"one_interfering_speaker ({os.path.basename(interfering_paths[0])})"
            
        elif condition == "three_interfering_speakers":
            # 3个干扰说话人
            if not interfering_paths or len(interfering_paths) < 3:
                n_given = len(interfering_paths) if interfering_paths else 0
                print(f"  警告: 需要3个干扰说话人，但只提供了{n_given}个")
                if n_given == 0:
                    noise = np.random.randn(len(clean_audio)) * 0.1
                    mixed = self.add_noise(clean_audio, noise, snr_db)
                    desc = "three_interfering_speakers (fallback: white_noise)"
                else:
                    # 用已有的干扰说话人
                    interference = np.zeros(len(clean_audio))
                    names = []
                    for p in interfering_paths[:3]:
                        audio = self.load_audio(p)
                        interference += librosa.util.normalize(audio)
                        names.append(os.path.basename(p))
                    mixed = self.add_noise(clean_audio, interference, snr_db)
                    desc = f"three_interfering_speakers ({', '.join(names)})"
            else:
                # 叠3个干扰说话人音频相加
                interference = np.zeros(len(clean_audio))
                names = []
                for p in interfering_paths[:3]:
                    audio = self.load_audio(p)
                    interference += librosa.util.normalize(audio)
                    names.append(os.path.basename(p))
                mixed = self.add_noise(clean_audio, interference, snr_db)
                desc = f"three_interfering_speakers ({', '.join(names)})"
        else:
            raise ValueError(f"不支持的测试条件: {condition}")
        
        return mixed, desc
    
    def _load_single_feature(self, video_path, feat_dir_name, encoder_name=None):
        """加载单个编码器的特征 (优先npy缓存，不存在时在线提取)"""
        feat_path = video_path.replace("/mp4/", f"/{feat_dir_name}/")
        feat_path = os.path.splitext(feat_path)[0] + ".npy"
        
        # 1. 优先加载已有npy
        if os.path.exists(feat_path):
            features = np.load(feat_path)
            if len(features.shape) == 3:  # (1, T, D) -> (T, D)
                features = features.squeeze(0)
            return features, feat_path, 'npy'
        
        # 2. 在线提取
        if self.extract_online and encoder_name:
            extractor = self._get_extractor(encoder_name)
            if extractor is not None:
                print(f"    在线提取 {encoder_name} 特征: {video_path}")
                try:
                    features = extractor.extract(video_path)
                    if features is not None:
                        if len(features.shape) == 3:
                            features = features.squeeze(0)
                        return features, video_path, 'online'
                except Exception as e:
                    print(f"    在线提取失败: {e}")
        
        return None, feat_path, 'missing'
    
    def _pad_features(self, features, target_frames=125):
        """填充/裁剪特征到目标帧数"""
        if features.shape[0] > target_frames:
            features = features[:target_frames]
        elif features.shape[0] < target_frames:
            pad = np.zeros((target_frames - features.shape[0], features.shape[-1]))
            features = np.vstack([features, pad])
        return features
    
    def load_visual_features(self, video_path, visual_feat_path=None):
        """
        加载视觉特征，支持单编码器和拼接/相加模式
        
        支持的编码器:
        - 单编码器: VSRiW, TalkNet, LoCoNet, AVHuBERT
        - 拼接模式: VSRiW_TalkNet_concatenate, AVHuBERT_TalkNet_concatenate, AVHuBERT_VSRiW_concatenate
        - 相加模式: VSRiW_TalkNet_addition
        """
        feat_dir_map = {
            'VSRiW': 'vsriw',
            'TalkNet': 'TalkNet_feats',
            'LoCoNet': 'Loconet_feats',
            'AVHuBERT': 'AVHuBERT_feats'
        }
        
        # 1. 如果直接指定了特征文件路径
        if visual_feat_path and os.path.exists(visual_feat_path):
            print(f"  加载视觉特征: {visual_feat_path}")
            features = np.load(visual_feat_path)
            if len(features.shape) == 3:
                features = features.squeeze(0)
            features = self._pad_features(features)
            return torch.tensor(features, dtype=torch.float32)
        
        # 2. 根据编码器类型加载特征
        encoder = self.visual_encoder
        
        # 拼接模式
        if "_concatenate" in encoder:
            parts = encoder.replace("_concatenate", "").split("_")
            enc1, enc2 = parts[0], parts[1]
            
            feat1, path1, src1 = self._load_single_feature(video_path, feat_dir_map[enc1], enc1)
            feat2, path2, src2 = self._load_single_feature(video_path, feat_dir_map[enc2], enc2)
            
            if feat1 is None or feat2 is None:
                missing = []
                if feat1 is None: missing.append(f"{enc1}: {path1}")
                if feat2 is None: missing.append(f"{enc2}: {path2}")
                print(f"  警告: 特征不可用:")
                for m in missing:
                    print(f"    - {m}")
                print(f"  使用零填充 (维度: {self.embedding_size})")
                return torch.zeros(125, self.embedding_size, dtype=torch.float32)
            
            feat1 = self._pad_features(feat1)
            feat2 = self._pad_features(feat2)
            features = np.concatenate([feat1, feat2], axis=1)
            print(f"  视觉特征 (拼接模式):")
            print(f"    - {enc1}: [{src1}] {feat1.shape}")
            print(f"    - {enc2}: [{src2}] {feat2.shape}")
            print(f"    - 拼接后: {features.shape}")
            
        # 相加模式
        elif "_addition" in encoder:
            parts = encoder.replace("_addition", "").split("_")
            enc1, enc2 = parts[0], parts[1]
            
            feat1, path1, src1 = self._load_single_feature(video_path, feat_dir_map[enc1], enc1)
            feat2, path2, src2 = self._load_single_feature(video_path, feat_dir_map[enc2], enc2)
            
            if feat1 is None or feat2 is None:
                missing = []
                if feat1 is None: missing.append(f"{enc1}: {path1}")
                if feat2 is None: missing.append(f"{enc2}: {path2}")
                print(f"  警告: 特征不可用:")
                for m in missing:
                    print(f"    - {m}")
                print(f"  使用零填充 (维度: {self.embedding_size})")
                return torch.zeros(125, self.embedding_size, dtype=torch.float32)
            
            feat1 = self._pad_features(feat1)
            feat2 = self._pad_features(feat2)
            features = feat1 + feat2
            print(f"  视觉特征 (相加模式):")
            print(f"    - {enc1}: [{src1}] {feat1.shape}")
            print(f"    - {enc2}: [{src2}] {feat2.shape}")
            print(f"    - 相加后: {features.shape}")
            
        # 单编码器模式
        else:
            feat_dir_name = feat_dir_map.get(encoder, 'vsriw')
            features, feat_path, src = self._load_single_feature(video_path, feat_dir_name, encoder)
            
            if features is None:
                print(f"  警告: 特征不可用: {feat_path}")
                print(f"  使用零填充 (维度: {self.embedding_size})")
                return torch.zeros(125, self.embedding_size, dtype=torch.float32)
            
            features = self._pad_features(features)
            print(f"  视觉特征: [{src}] {features.shape}")
        
        return torch.tensor(features, dtype=torch.float32)
    
    @torch.no_grad()
    def compute_metrics(self, clean_audio, noisy_audio, enhanced_audio):
        """
        计算性能指标
        
        Returns:
            dict: 包含各项指标的字典
        """
        clean_t = torch.tensor(clean_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        noisy_t = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        enhanced_t = torch.tensor(enhanced_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        results = {}
        
        # SNR
        try:
            snr_enhanced = self.metrics['snr'](enhanced_t, clean_t).item()
            snr_noisy = self.metrics['snr'](noisy_t, clean_t).item()
            results['snr'] = snr_enhanced
            results['snr_gain'] = snr_enhanced - snr_noisy
        except Exception as e:
            results['snr'] = float('nan')
            results['snr_gain'] = float('nan')
        
        # SI-SDR
        try:
            sisdr_enhanced = self.metrics['sisdr'](enhanced_t, clean_t).item()
            sisdr_noisy = self.metrics['sisdr'](noisy_t, clean_t).item()
            results['sisdr'] = sisdr_enhanced
            results['sisdr_gain'] = sisdr_enhanced - sisdr_noisy
        except Exception as e:
            results['sisdr'] = float('nan')
            results['sisdr_gain'] = float('nan')
        
        # PESQ
        try:
            results['pesq'] = self.metrics['pesq'](enhanced_t, clean_t).item()
        except Exception as e:
            results['pesq'] = float('nan')
        
        # ESTOI
        try:
            results['estoi'] = self.metrics['estoi'](enhanced_t, clean_t).item()
        except Exception as e:
            results['estoi'] = float('nan')
        
        return results
    
    def _print_metrics(self, metrics, prefix="  "):
        """打印指标"""
        print(f"{prefix}📊 性能指标:")
        print(f"{prefix}   SNR:  {metrics['snr']:.4f} dB  (Gain: {metrics['snr_gain']:+.4f} dB)")
        print(f"{prefix}   SISDR: {metrics['sisdr']:.4f} dB  (Gain: {metrics['sisdr_gain']:+.4f} dB)")
        print(f"{prefix}   PESQ:  {metrics['pesq']:.4f}")
        print(f"{prefix}   ESTOI: {metrics['estoi']:.4f}")

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
    
    def process_file(self, input_path, output_dir, condition="noise_only", snr_db=0, 
                     noise_path=None, interfering_paths=None, visual_feat_path=None):
        """
        处理单个视频/音频文件
        
        Args:
            input_path: 输入MP4或音频文件
            output_dir: 输出目录
            condition: 测试条件 (noise_only/one_interfering_speaker/three_interfering_speakers)
            snr_db: 信噪比 (dB)
            noise_path: 噪声文件路径
            interfering_paths: 干扰说话人MP4路径列表
            visual_feat_path: 视觉特征文件路径
        
        Returns:
            dict or None: 性能指标字典，失败时返回None
        """
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 加载干净音频
            print(f"  提取音频...")
            clean_audio = self.load_audio(input_path)
            clean_audio = librosa.util.normalize(clean_audio)
            
            # 2. 根据测试条件混合音频
            print(f"  混合音频 (condition={condition}, SNR={snr_db}dB)...")
            noisy_audio, cond_desc = self.mix_audio(
                clean_audio, condition, snr_db, noise_path, interfering_paths
            )
            print(f"  条件: {cond_desc}")
            
            # 3. 加载视觉特征
            print(f"  加载视觉特征...")
            visual_feat = self.load_visual_features(input_path, visual_feat_path)
            
            # 4. 增强
            print(f"  增强处理...")
            enhanced_audio = self.enhance(noisy_audio, visual_feat)
            
            # 5. 计算性能指标
            print(f"  计算指标...")
            metrics = self.compute_metrics(clean_audio, noisy_audio, enhanced_audio)
            self._print_metrics(metrics)
            
            # 6. 保存结果
            tag = f"{condition}_snr{snr_db}"
            clean_path = os.path.join(output_dir, f"{base_name}_clean.wav")
            noisy_path = os.path.join(output_dir, f"{base_name}_noisy_{tag}.wav")
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced_{tag}.wav")
            
            sf.write(clean_path, clean_audio, self.sample_rate)
            sf.write(noisy_path, noisy_audio, self.sample_rate)
            sf.write(enhanced_path, enhanced_audio, self.sample_rate)
            
            print(f"  ✅ 保存:")
            print(f"     干净音频: {clean_path}")
            print(f"     带噪音频: {noisy_path}")
            print(f"     增强音频: {enhanced_path}")
            
            return metrics
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_directory(self, input_dir, output_dir, condition="noise_only", snr_db=0,
                          noise_path=None, interfering_paths=None):
        """处理目录中的所有视频文件"""
        video_files = glob(os.path.join(input_dir, '**', '*.mp4'), recursive=True)
        print(f"找到 {len(video_files)} 个视频文件")
        
        all_metrics = []
        success_count = 0
        for video_path in tqdm(video_files, desc="处理中"):
            print(f"\n处理: {video_path}")
            rel_path = os.path.relpath(video_path, input_dir)
            sub_output_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            
            metrics = self.process_file(
                video_path, sub_output_dir, condition, snr_db, noise_path, interfering_paths
            )
            if metrics is not None:
                success_count += 1
                all_metrics.append(metrics)
        
        # 汇总统计
        print(f"\n{'='*60}")
        print(f"处理完成! 成功: {success_count}/{len(video_files)}")
        
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
                avg_metrics[key] = np.mean(vals) if vals else float('nan')
            
            print(f"\n📊 平均性能指标 ({len(all_metrics)} 个文件):")
            print(f"   SNR:   {avg_metrics['snr']:.4f} dB  (Gain: {avg_metrics['snr_gain']:+.4f} dB)")
            print(f"   SISDR: {avg_metrics['sisdr']:.4f} dB  (Gain: {avg_metrics['sisdr_gain']:+.4f} dB)")
            print(f"   PESQ:  {avg_metrics['pesq']:.4f}")
            print(f"   ESTOI: {avg_metrics['estoi']:.4f}")
            print(f"{'='*60}")
            
            # 写入日志文件
            log_path = os.path.join(output_dir, "results.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a") as f:
                f.write(f"\n[{timestamp}]\n")
                f.write(f"  visual_encoder: {self.visual_encoder}\n")
                f.write(f"  input_dir: {input_dir}\n")
                f.write(f"  condition: {condition}\n")
                f.write(f"  snr: {snr_db}\n")
                f.write(f"  noise: {noise_path or 'N/A'}\n")
                f.write(f"  interfering: {interfering_paths or 'N/A'}\n")
                f.write(f"  total: {len(video_files)}, success: {success_count}\n")
                f.write(f"  AVG SNR: {avg_metrics['snr']:.4f}, SNR Gain: {avg_metrics['snr_gain']:.4f}\n")
                f.write(f"  AVG SISDR: {avg_metrics['sisdr']:.4f}, SISDR Gain: {avg_metrics['sisdr_gain']:.4f}\n")
                f.write(f"  AVG PESQ: {avg_metrics['pesq']:.4f}\n")
                f.write(f"  AVG ESTOI: {avg_metrics['estoi']:.4f}\n")
            print(f"\n日志已保存: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="音视频语音增强推理脚本")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入MP4视频文件或目录")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--ckpt_path", "-c", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--visual_encoder", "-v", type=str, default="VSRiW_TalkNet_concatenate",
                        choices=["VSRiW", "TalkNet", "LoCoNet", "AVHuBERT", 
                                 "VSRiW_TalkNet_concatenate", "VSRiW_TalkNet_addition",
                                 "AVHuBERT_TalkNet_concatenate", "AVHuBERT_VSRiW_concatenate"],
                        help="视觉编码器类型")
    parser.add_argument("--snr", type=int, default=0,
                        help="信噪比 (dB), 默认0")
    parser.add_argument("--condition", type=str, default="noise_only",
                        choices=["noise_only", "one_interfering_speaker", "three_interfering_speakers"],
                        help="测试条件")
    parser.add_argument("--noise", type=str, default=None,
                        help="噪声文件路径 (noise_only时使用，不提供则白噪声)")
    parser.add_argument("--interfering", type=str, nargs='+', default=None,
                        help="干扰说话人MP4文件路径 (1个或3个)")
    parser.add_argument("--visual_feat", type=str, default=None,
                        help="视觉特征文件路径 (可选)")
    parser.add_argument("--no_extract", action="store_true",
                        help="禁用在线特征提取，仅使用已有npy")
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
        ckpt_path=args.ckpt_path,
        visual_encoder=args.visual_encoder,
        embedding_size=args.embedding_size,
        device=args.device,
        extract_online=not args.no_extract
    )
    
    print(f"\n配置:")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  条件: {args.condition}")
    print(f"  SNR: {args.snr} dB")
    if args.condition == "noise_only":
        print(f"  噪声: {args.noise or '白噪声'}")
    else:
        print(f"  干扰说话人: {args.interfering or '未指定'}")
    print(f"  在线提取: {'关闭' if args.no_extract else '开启 (优先用npy缓存)'}")
    print()
    
    # 处理输入
    if os.path.isfile(args.input):
        print(f"处理文件: {args.input}")
        metrics = enhancer.process_file(
            args.input, args.output, args.condition, args.snr,
            args.noise, args.interfering, args.visual_feat
        )
        if metrics:
            # 单文件也写log
            log_path = os.path.join(args.output, "results.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a") as f:
                f.write(f"\n[{timestamp}]\n")
                f.write(f"  visual_encoder: {args.visual_encoder}\n")
                f.write(f"  input: {args.input}\n")
                f.write(f"  condition: {args.condition}\n")
                f.write(f"  snr: {args.snr}\n")
                f.write(f"  noise: {args.noise or 'N/A'}\n")
                f.write(f"  interfering: {args.interfering or 'N/A'}\n")
                f.write(f"  SNR: {metrics['snr']:.4f}, SNR Gain: {metrics['snr_gain']:.4f}\n")
                f.write(f"  SISDR: {metrics['sisdr']:.4f}, SISDR Gain: {metrics['sisdr_gain']:.4f}\n")
                f.write(f"  PESQ: {metrics['pesq']:.4f}\n")
                f.write(f"  ESTOI: {metrics['estoi']:.4f}\n")
            print(f"\n日志已保存: {log_path}")
    elif os.path.isdir(args.input):
        print(f"处理目录: {args.input}")
        enhancer.process_directory(
            args.input, args.output, args.condition, args.snr,
            args.noise, args.interfering
        )
    else:
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] = "2"
    main()
