import os
import numpy as np
import torch
from torch.utils import data
import pandas as pd
import librosa
import hashlib
import random
from utils.utils import crop_pad_audio
from utils.augment_visual import augment_visual
import config as config


SPLIT_FILE_PATH = "./data/split.csv"
SAMPLING_RATE = config.SAMPLING_RATE


class VoxCeleb2(data.Dataset):
    
    def __init__(self, split, data_path, visual_encoder, embedding_size, all_data=None, condition=None, snr=None):
        self.split = split
        self.data_path = data_path
        if all_data is None:
            all_data = pd.read_csv(SPLIT_FILE_PATH)
        
        # 根据 dataset 列确定数据集名称
        if "dataset" in all_data.columns:
            datasets = all_data["dataset"].unique()
            if len(datasets) == 1:
                self.dataset_name = datasets[0]
            else:
                # 多个数据集混合时，默认使用第一个
                self.dataset_name = datasets[0]
                print(f"Warning: Multiple datasets found {datasets}, using '{self.dataset_name}'")
        else:
            # 兼容旧格式，默认为 VoxCeleb2
            self.dataset_name = "VoxCeleb2"
        
        musan_fps = pd.read_csv("./data/musan_split.csv")
        self.musan_fps = musan_fps[musan_fps["split"] == self.split]
        if self.split == "train":
            self.data = all_data[all_data["split"] == self.split]["audio_fp"]
            # 过滤失败样本
            self._filter_failed_samples(visual_encoder, data_path)
        elif self.split == "val":
            # 对于非 VoxCeleb2 数据集，从 split.csv 动态采样
            if self.dataset_name != "VoxCeleb2":
                val_data = all_data[all_data["split"] == "val"]
                # 取前1000个（完全确定性，保证可复现）
                if len(val_data) > 1000:
                    val_data = val_data.head(1000)
                self.data = val_data["audio_fp"].reset_index(drop=True)
            else:
                # VoxCeleb2 保持原有逻辑
                self.data = pd.read_csv("./data/VoxCeleb2_val_1000_fps.txt", header=None)[0]
            # val 也需要过滤失败样本
            self._filter_failed_samples(visual_encoder, data_path)
        
        elif self.split == "test":
            # 对于非 VoxCebeb2 数据集，从 split.csv 动态采样
            if self.dataset_name != "VoxCeleb2":
                test_data = all_data[all_data["split"] == "test"]
                # 取前1000个（完全确定性，保证可复现）
                if len(test_data) > 1000:
                    test_data = test_data.head(1000)
                self.data = test_data["audio_fp"].reset_index(drop=True)
            else:
                # VoxCeleb2 保持原有逻辑
                self.data = pd.read_csv("./data/VoxCeleb2_test_1000_fps.txt", header=None)[0]
            self.condition = condition
            self.snr = snr
            # test 也需要过滤失败样本
            self._filter_failed_samples(visual_encoder, data_path)
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        
        self.embedding_path_dict = {
            "VSRiW": "/vsriw/",
            "TalkNet": "/TalkNet_feats/",
            "Loconet": "/Loconet_feats/",
            "AVHuBERT": "/AVHuBERT_feats/",
            "VSRIW_LRS3": "/vsriw_lrs3/",
            
        }
        # 各编码器的真实维度，用于零填充
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
        audio_fp_raw = self.data.iloc[idx]
        
        # 根据数据集名称构建正确的音频路径
        if self.dataset_name == "VoxCeleb2" or self.dataset_name == "":
            # VoxCeleb2 格式: audio_fp 是相对路径，如 dev/aac/id00012/xxx.m4a
            audio_fp = os.path.join(self.data_path, audio_fp_raw)
            mixed_audio_fp = audio_fp.replace("/aac/", "/mixed_wav/").replace(".m4a", ".wav")
        else:
            # 其他数据集（包括 ChineseLips）: audio_fp 是相对路径，如 train/wav/xxx.wav
            audio_fp = os.path.join(self.data_path, audio_fp_raw)
            # mixed 音频路径: train/mixed_wav/xxx.wav
            mixed_audio_fp = audio_fp.replace("/wav/", "/mixed_wav/").replace(".wav", ".wav")
        
        # get raw target audio
        audio = crop_pad_audio(audio_fp, SAMPLING_RATE, 5)
        input_audio = librosa.util.normalize(audio)
        
        if self.split == "test":
            # 根据数据集构建测试 mixed 音频路径
            if self.dataset_name == "VoxCeleb2" or self.dataset_name == "":
                mixed_audio_fp = mixed_audio_fp.replace("/mixed_wav/", f"/mixed_wav/{self.condition}/{self.snr}/")
            else:
                # 其他数据集（包括 ChineseLips）: test/mixed_wav/{condition}/{snr}/
                base_dir = os.path.dirname(audio_fp_raw)  # 如: test/wav
                split_name = base_dir.split("/")[0]       # 如: test
                mixed_audio_fp = audio_fp.replace("/wav/", f"/mixed_wav/{self.condition}/{self.snr}/")
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)

        else:
            mixed_audio = crop_pad_audio(mixed_audio_fp, SAMPLING_RATE, 5)
        input_audio, mixed_audio = torch.tensor(input_audio), torch.tensor(mixed_audio)
        if_speaker_fp = "None"
        
        
        
        
        face_embed = torch.zeros((125, self.embedding_size))
        # get face embeddings
        if "_" in self.visual_encoder: 
            
            if "addition" in self.visual_encoder:
                face_embed = self._add_embeddings(audio_fp, self.visual_encoder)
            elif "concatenate" in self.visual_encoder:
                face_embed = self._concatenate_embeddings(audio_fp, self.visual_encoder)
            
        else:
            # 根据数据集动态构建特征文件路径
            if self.dataset_name == "VoxCeleb2" or self.dataset_name == "":
                # VoxCeleb2 格式: /aac/ -> 特征目录
                fe_fp = audio_fp.replace("/aac/", self.embedding_path_dict[self.visual_encoder]).replace(".m4a", ".npy")
            else:
                # 其他数据集（包括 ChineseLips）: /wav/ -> 特征目录
                fe_fp = audio_fp.replace("/wav/", self.embedding_path_dict[self.visual_encoder]).replace(".wav", ".npy")
            try:
                face_embed = np.load(fe_fp, mmap_mode="r", allow_pickle=True)
                # crop and pad face embeddings to 5 seconds
                face_embed = self._crop_pad_face_embeddings(face_embed)
                face_embed = torch.tensor(face_embed)
            except (FileNotFoundError, OSError, ValueError) as e:
                print(f"Warning: Failed to load embedding {fe_fp}, using zeros. Error: {e}")
                # 使用零填充作为后备
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

    def _concatenate_embeddings(self, audio_fp, combined_features):
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
            
        # 根据数据集动态构建特征文件路径
        if self.dataset_name == "VoxCeleb2" or self.dataset_name == "":
            # VoxCeleb2 格式: /aac/ -> 特征目录
            fe_fp1 = audio_fp.replace("/aac/", self.embedding_path_dict[fe1]).replace(".m4a", ".npy")
            fe_fp2 = audio_fp.replace("/aac/", self.embedding_path_dict[fe2]).replace(".m4a", ".npy")
        else:
            # 其他数据集（包括 ChineseLips）: /wav/ -> 特征目录
            fe_fp1 = audio_fp.replace("/wav/", self.embedding_path_dict[fe1]).replace(".wav", ".npy")
            fe_fp2 = audio_fp.replace("/wav/", self.embedding_path_dict[fe2]).replace(".wav", ".npy")

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

    def _filter_failed_samples(self, visual_encoder, data_path):
        """过滤提取失败的样本"""
        encoder_failed_files = {
            "VSRiW": "failed_VSRiW_frontE_feat_split.txt",
            "TalkNet": "failed_TalkNet_frontE_feat_split.txt",
            "Loconet": "failed_Loconet_frontE_feat_split.txt",
            "AVHuBERT": "failed_avhubert_frontE_feat_split.txt",
        }
        # 找出当前配置需要检查的编码器
        encoders_to_check = []
        for enc_name, failed_file in encoder_failed_files.items():
            if enc_name in visual_encoder:
                encoders_to_check.append((enc_name, failed_file))
        
        for enc_name, failed_file in encoders_to_check:
            failed_fp = os.path.join(data_path, failed_file)
            if os.path.exists(failed_fp) and os.path.getsize(failed_fp) > 0:
                print(f"  Loading {enc_name} failed sample filtering file: {failed_fp}")
                failed_raw = pd.read_csv(failed_fp, header=None, comment='#')[0]
                print(f"    Failed file has {len(failed_raw)} entries")
                if len(failed_raw) > 0:
                    print(f"    Sample failed path (raw): {failed_raw.iloc[0]}")
                # 将 failed 路径转为相对路径并根据数据集转换格式
                failed = failed_raw.str.strip()
                failed = failed.str.replace(data_path + "/", "", regex=False)
                
                if self.dataset_name == "VoxCeleb2" or self.dataset_name == "":
                    # VoxCeleb2 格式: mp4 -> aac, .mp4 -> .m4a
                    failed = failed.str.replace("/mp4/", "/aac/").str.replace(".mp4", ".m4a")
                else:
                    # 其他数据集（包括 ChineseLips）: mp4 -> wav, .mp4 -> .wav
                    failed = failed.str.replace("/mp4/", "/wav/").str.replace(".mp4", ".wav")
                if len(failed) > 0:
                    print(f"    Sample failed path (converted): {failed.iloc[0]}")
                before_count = len(self.data)
                self.data = self.data[~self.data.isin(failed)]
                after_count = len(self.data)
                print(f"    Filtered out {before_count - after_count} failed {enc_name} samples (remaining: {after_count})")
            else:
                print(f"  {enc_name} failed file not found or empty: {failed_fp} (skipping)")
    
    def _add_embeddings(self, audio_fp, combined_features):
        fe1, fe2 = combined_features.split("_")[0], combined_features.split("_")[1]
        fe_fp1 = audio_fp.replace("/aac/", self.embedding_path_dict[fe1]).replace(".m4a", ".npy")
        fe_fp2 = audio_fp.replace("/aac/", self.embedding_path_dict[fe2]).replace(".m4a", ".npy")

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
        
        
    

    
    
    