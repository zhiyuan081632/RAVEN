import os
import numpy as np
import torch
from torch.utils import data
import pandas as pd
import librosa
import random
from utils.utils import crop_pad_audio
from utils.augment_visual import augment_visual
import config as config

SAMPLING_RATE = config.SAMPLING_RATE


def _load_list_files(file_paths):
    """从一个或多个 list 文件加载路径列表（每行一个绝对路径）。
    自动跳过 header 行（如 'filepath'）和空行。
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    all_paths = []
    for fp in file_paths:
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line == 'filepath':
                    continue
                all_paths.append(line)
    return all_paths


def _mp4_to_wav(mp4_path):
    """mp4 绝对路径 -> wav 绝对路径"""
    return mp4_path.replace('/mp4/', '/wav/').replace('.mp4', '.wav')


def _mp4_to_feat(mp4_path, feat_dir):
    """mp4 绝对路径 -> 特征 npy 绝对路径
    feat_dir 如 '/vsriw/', '/TalkNet_feats/' 等
    """
    return mp4_path.replace('/mp4/', feat_dir).replace('.mp4', '.npy')


class VoxCeleb2(data.Dataset):
    """基于 mp4 绝对路径 list 文件的数据集。
    
    speech_lists: 语音 list 文件路径列表，每行一个 mp4 绝对路径
    noise_lists:  噪声 list 文件路径列表 (musan_noise_*.txt)
    music_lists:  音乐 list 文件路径列表 (musan_music_*.txt)
    """
    
    def __init__(self, split, visual_encoder, embedding_size,
                 speech_lists, noise_lists=None, music_lists=None,
                 condition=None, snr=None):
        self.split = split
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        
        # 加载 speech mp4 路径列表
        self.data = _load_list_files(speech_lists)
        print(f"  [{split}] Loaded {len(self.data)} speech files from {speech_lists}")
        
        # 加载噪声
        self.noise_fps = _load_list_files(noise_lists) if noise_lists else []
        self.music_fps = _load_list_files(music_lists) if music_lists else []
        
        # test 专用参数
        self.condition = condition
        self.snr = snr
        
        self.embedding_path_dict = {
            "VSRiW": "/feats_VSRiW/",
            "TalkNet": "/feats_TalkNet/",
            "Loconet": "/feats_Loconet/",
            "AVHuBERT": "/feats_AVHuBERT/",
            "VSRIW_LRS3": "/feats_VSRiW_lrs3/",
        }
        self.encoder_dim_dict = {
            "VSRiW": 512,
            "TalkNet": 512,
            "Loconet": 512,
            "AVHuBERT": 768,
            "VSRIW_LRS3": 512,
        }
        self.device = "cuda"
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Skipping idx={idx}, error: {e}")
            return None

    def _load_item(self, idx):
        mp4_fp = self.data[idx]  # mp4 绝对路径
        audio_fp = _mp4_to_wav(mp4_fp)  # wav 绝对路径
        
        # mixed 音频路径: /wav/ -> /mixed_wav/
        mixed_audio_fp = audio_fp.replace('/wav/', '/mixed_wav/')
        
        # get raw target audio
        audio = crop_pad_audio(audio_fp, SAMPLING_RATE, 5)
        input_audio = librosa.util.normalize(audio)
        
        if self.split == "test":
            mixed_audio_fp = audio_fp.replace('/wav/', f'/mixed_wav/{self.condition}/{self.snr}/')
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)
        else:
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)
        input_audio, mixed_audio = torch.tensor(input_audio), torch.tensor(mixed_audio)
        if_speaker_fp = "None"
        
        
        
        
        face_embed = torch.zeros((125, self.embedding_size))
        # get face embeddings
        if "_" in self.visual_encoder:
            if "addition" in self.visual_encoder:
                face_embed = self._add_embeddings(mp4_fp, self.visual_encoder)
            elif "concatenate" in self.visual_encoder:
                face_embed = self._concatenate_embeddings(mp4_fp, self.visual_encoder)
        else:
            fe_fp = _mp4_to_feat(mp4_fp, self.embedding_path_dict[self.visual_encoder])
            try:
                face_embed = np.load(fe_fp, mmap_mode="r", allow_pickle=True)
                face_embed = self._crop_pad_face_embeddings(face_embed)
                face_embed = torch.tensor(face_embed)
            except (FileNotFoundError, OSError, ValueError) as e:
                print(f"Warning: Failed to load embedding {fe_fp}, using zeros. Error: {e}")
                face_embed = torch.zeros((125, self.embedding_size))
            if self.split != "test":
                face_embed = augment_visual(face_embed, self.visual_encoder)[0]

        
        return {
            "face_embed": face_embed,
            "input_audio": input_audio,
            "mixed_audio": mixed_audio,
            "audio_fp": audio_fp,
            "interfering_speaker_fp": if_speaker_fp
        }

    def _concatenate_embeddings(self, mp4_fp, combined_features):
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        fe_fp1 = _mp4_to_feat(mp4_fp, self.embedding_path_dict[fe1])
        fe_fp2 = _mp4_to_feat(mp4_fp, self.embedding_path_dict[fe2])

        # 使用各编码器的真实维度
        dim1 = self.encoder_dim_dict.get(fe1, 512)
        dim2 = self.encoder_dim_dict.get(fe2, 512)
        
        # 分别加载每个编码器的 embedding
        try:
            embed1 = np.load(fe_fp1, mmap_mode="r", allow_pickle=True)
            embed1 = self._crop_pad_face_embeddings(embed1)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Warning: Failed to load {fe_fp1}, using zeros ({dim1}d). Error: {e}")
            embed1 = np.zeros((125, dim1), dtype=np.float32)

        try:
            embed2 = np.load(fe_fp2, mmap_mode="r", allow_pickle=True)
            embed2 = self._crop_pad_face_embeddings(embed2)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Warning: Failed to load {fe_fp2}, using zeros ({dim2}d). Error: {e}")
            embed2 = np.zeros((125, dim2), dtype=np.float32)

        face_embed1, face_embed2 = torch.tensor(embed1), torch.tensor(embed2)

        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        return torch.cat((face_embed1, face_embed2), dim=1)

    # _filter_failed_samples 已移除，不再需要
    # 如需过滤，直接在 list 文件中剔除对应条目即可
    
    def _add_embeddings(self, mp4_fp, combined_features):
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        fe_fp1 = _mp4_to_feat(mp4_fp, self.embedding_path_dict[fe1])
        fe_fp2 = _mp4_to_feat(mp4_fp, self.embedding_path_dict[fe2])

        # 使用各编码器的真实维度（addition 模式要求维度相同）
        dim1 = self.encoder_dim_dict.get(fe1, 512)
        dim2 = self.encoder_dim_dict.get(fe2, 512)
        # addition 模式只支持维度相同的编码器
        assert dim1 == dim2, f"Addition mode requires same dimensions, got {fe1}={dim1} and {fe2}={dim2}"
        
        # 分别加载每个编码器的 embedding
        try:
            embed1 = np.load(fe_fp1, mmap_mode="r")
            embed1 = self._crop_pad_face_embeddings(embed1)
        except (FileNotFoundError, OSError, ValueError):
            embed1 = np.zeros((125, dim1), dtype=np.float32)

        try:
            embed2 = np.load(fe_fp2, mmap_mode="r")
            embed2 = self._crop_pad_face_embeddings(embed2)
        except (FileNotFoundError, OSError, ValueError):
            embed2 = np.zeros((125, dim2), dtype=np.float32)

        face_embed1, face_embed2 = torch.tensor(embed1), torch.tensor(embed2)
        
        if self.split != "test":
            face_embed1 = augment_visual(face_embed1, fe1)[0]
            face_embed2 = augment_visual(face_embed2, fe2)[0]

        return face_embed1 + face_embed2
    
    
    def normalize(self, x, norm='l2'):
        if norm == 'l2':
            return self.l2_normalize(x)
        elif norm == 'z_score':
            return self.z_score_normalization(x)
        else:
            return
    
    def l2_normalize(self, x):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)
    
    def z_score_normalization(self, x):
        mean = x.mean(dim=1, keepdim=True)  
        std = x.std(dim=1, keepdim=True) + 1e-6  
        return (x - mean) / std
    


    
    def _crop_pad_face_embeddings(self, fe):
        
        if len(fe) < 25*5:
            fe = np.pad(fe, ((0, 25*5 - len(fe)), (0,0)))
        fe = fe[:25*5]
        
        return fe
        
        
    

    
    
    