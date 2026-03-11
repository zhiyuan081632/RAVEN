
import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sys.path.append(config.TALKNET_PATH)
from talkNet import talkNet
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from torchvision.transforms.functional import center_crop
from pathlib import Path

from torchvision.transforms import ToPILImage

SPEECH_FOLDER_PATH = config.SPEECH_FOLDER_PATH



class VideoDataset(Dataset):
    def __init__(
            self, 
            data_split_path ,
            data_path,
            split=None,
            dataset=None,
            ) :
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dataset = dataset
        # 根据 split 和 dataset 过滤数据
        all_data = pd.read_csv(data_split_path)
        if split:
            all_data = all_data[all_data["split"] == split]
        if dataset:
            all_data = all_data[all_data["dataset"] == dataset]
            print(f"Filtered by split='{split}', dataset='{dataset}': {len(all_data)} samples")
        # 使用 video_fp 列，避免 aac/wav 路径差异
        self.file_list = all_data["video_fp"].reset_index(drop=True).tolist()


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        fp = os.path.join(self.data_path, self.file_list[index])
        video_tensor_orig = torchvision.io.read_video(fp, pts_unit='sec')[0]
        video_tensor = random_5s_clip(video_tensor_orig)
        assert video_tensor.shape[0] == 125, f"Video {fp} has {video_tensor.shape} frames"
        
        rgb_to_grayscale_weight = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
        rgb_to_grayscale_weight = rgb_to_grayscale_weight.view(1, 1, 1, 3)
        gray_video = torch.sum(video_tensor * rgb_to_grayscale_weight, dim=-1, keepdim=False)
        gray_video = center_crop(gray_video, [112, 112])
        
        return gray_video, fp


class TalkNetBatchedPreprocessing:
    def __init__(
            self,
            model_path,
            data_split_path,
            data_path,
            split,
            dataset=None,
            batch_size: int = None,
            num_workers: int = 1,

            device = "cuda"
        ) -> None:
        super().__init__()
        self.split = split
        self.dataset = dataset
        self.ds = VideoDataset(data_split_path, data_path, split = self.split, dataset=self.dataset)
        self.dataloader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
        self.model = talkNet().to(device)
        self.model.loadParameters(model_path)
        self.device = device

    def extract_features(self):
        # 使用统一的失败记录文件（不分 train/val/test）
        failed_text_path = os.path.join(config.SPEECH_FOLDER_PATH, "failed_TalkNet_frontE_feat_split.txt")
        success_count = 0
        skip_count = 0
        failure_count = 0
        total = len(self.ds)
        
        with open(failed_text_path, 'w') as failed_txt:
            for video_tensor, fp in tqdm(self.dataloader):
                try:
                    if video_tensor is not None:
                        output_path = fp.replace("/mp4/", "/TalkNet_feats/")
                        output_ft_path = output_path.replace(".mp4", ".npy")
                        # 跳过已存在的npy文件
                        if os.path.exists(output_ft_path):
                            skip_count += 1
                            continue
                        video_tensor = video_tensor.unsqueeze(0)
                        B,T,H,W = video_tensor.shape
                        video_tensor = video_tensor.to(self.device)
                        video_tensor = video_tensor.view(B*T, 1, 1, W, H)
                        video_tensor = (video_tensor / 255 - 0.4161) / 0.1688
                        
                        video_embed = self.model.model.visualFrontend(video_tensor)
                        video_embed = video_embed.view(B, T, -1)
                        video_embed = video_embed.squeeze(0)

                        if not os.path.exists(os.path.dirname(output_ft_path)):
                            os.makedirs(os.path.dirname(output_ft_path))

                        video_embed = video_embed.detach().cpu().numpy()
                        
                        assert video_embed.shape == (125, 512), f"Video {fp} has shape {video_embed.shape}"
                        
                        np.save(output_ft_path, video_embed)
                        success_count += 1
                    else:
                        print(f"Video {fp} preprocessing failed.")
                        failed_txt.write(fp + '\n')
                        failure_count += 1
                except Exception as e:
                    print(f"Error processing {fp}: {e}")
                    failed_txt.write(fp + f" # {str(e)}\n")
                    failure_count += 1
        
        print(f"\n=== TalkNet 特征提取统计 ===")
        print(f"总文件数: {total}")
        print(f"新生成: {success_count}")
        print(f"已跳过: {skip_count}")
        print(f"失败: {failure_count}")
    
    def test_zero(self):
        
        with torch.no_grad():

            video_tensor = torch.zeros(1, 112, 112)
            video_tensor = video_tensor.unsqueeze(0)
            B,T,H,W = video_tensor.shape
            video_tensor = video_tensor.to(self.device)
            video_tensor = video_tensor.view(B*T, 1, 1, W, H)
            
            video_embed = self.model.model.visualFrontend(video_tensor)
            video_embed = video_embed.view(B, T, -1)
            video_embed = video_embed.squeeze(0)
            
            np.save(f"./pretrained_zeros/TalkNet_zero_oneframe.npy", video_embed.detach().cpu().numpy())
        
            print(video_embed.shape)
            print(video_embed)
        


def random_5s_clip(video, frame_rate=25):

    T, H, W, C = video.shape
    target_frames = 5 * frame_rate 

    if T >= target_frames:
        
        return video[:target_frames, :, :, :]
    else:
        
        pad_length = target_frames - T
        padding = (0, 0,  
                   0, 0,  
                   0, 0,  
                   0, pad_length)  
        
        return F.pad(video, padding, mode='constant', value=0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract TalkNet visual features")
    parser.add_argument("--speech_dataset", type=str, default="VoxCeleb2",
                        help="Speech dataset to process")
    parser.add_argument("--split", type=str, default="",
                        help="Split to process: train, val, test")
    args = parser.parse_args()
    
    model_path = os.path.join(config.TALKNET_PATH, "pretrain_TalkSet.model")
    
    # Download pretrained model if not exists
    if not os.path.isfile(model_path):
        print(f"Model not found at {model_path}, downloading...")
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = f"gdown --id {Link} -O {model_path}"
        import subprocess
        subprocess.call(cmd, shell=True, stdout=None)
        print(f"Model downloaded to {model_path}")
    
    data_split = "./data/split.csv"
    data_path = config.SPEECH_DATASETS.get(args.speech_dataset, config.SPEECH_FOLDER_PATH)
    print(f"Using dataset: {args.speech_dataset} -> {data_path}")
    
    process = TalkNetBatchedPreprocessing(
        model_path,
        data_split,
        data_path,
        split=args.split,
        dataset=args.speech_dataset,
        num_workers=2 # os.cpu_count() = 16 or 64
    )
    process.extract_features()

if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"
    main()

    