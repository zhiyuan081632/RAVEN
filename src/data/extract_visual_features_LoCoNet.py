
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

# Add LoCoNet path for imports
sys.path.insert(0, config.LOCONET_PATH)
sys.path.insert(0, os.path.join(config.LOCONET_PATH, 'dlhammer'))
from loconet import Loconet, loconet
from dlhammer import bootstrap
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from torchvision.transforms.functional import center_crop
from pathlib import Path

from torchvision.transforms import ToPILImage





FEAT_NAME = "feats_LoCoNet"

def _load_list_files(file_paths):
    """从一个或多个 list 文件加载路径列表"""
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



class VideoDataset(Dataset):
    def __init__(
            self, 
            data_path=None,
            file_list_paths=None,
            ) :
        super().__init__()
        self.data_path = data_path
        if file_list_paths is not None:
            self.file_list = _load_list_files(file_list_paths)
        elif data_path is not None:
            self.file_list = sorted(str(p) for p in Path(data_path).rglob("*.mp4"))
        else:
            raise ValueError("Either data_path or file_list_paths must be provided")
        print(f"Found {len(self.file_list)} video files")


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        fp = self.file_list[index]
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
            data_path=None,
            output_dir=None,
            file_list_paths=None,
            batch_size: int = None,
            num_workers: int = 1,
            device = "cpu"
        ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.list_mode = file_list_paths is not None
        self.ds = VideoDataset(data_path=data_path, file_list_paths=file_list_paths)
        self.dataloader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
        # Initialize cfg using bootstrap with config file
        config_file = os.path.join(config.LOCONET_PATH, 'configs/multi.yaml')
        # Set sys.argv to pass config file to bootstrap
        original_argv = sys.argv.copy()
        sys.argv = ['script_name', '--cfg', config_file]
        cfg = bootstrap(print_cfg=False)
        sys.argv = original_argv  # Restore original argv
        self.model = loconet(cfg).to(device)
        self.model.loadParameters(model_path)
        self.device = device

    def _get_output_path(self, fp):
        if self.list_mode:
            return fp.replace('/mp4/', f'/{FEAT_NAME}/').replace('.mp4', '.npy')
        else:
            rel_path = os.path.relpath(fp, self.ds.data_path)
            return os.path.join(self.output_dir, rel_path).replace('.mp4', '.npy')

    def extract_features(self):
        if self.list_mode:
            first_out = self._get_output_path(self.ds.file_list[0])
            output_root = first_out[:first_out.find(f'/{FEAT_NAME}/') + len(f'/{FEAT_NAME}')]
        else:
            output_root = self.output_dir
        os.makedirs(output_root, exist_ok=True)
        failed_text_path = os.path.join(output_root, f"failed_{FEAT_NAME}.txt")
        success_count = 0
        skip_count = 0
        failure_count = 0
        total = len(self.ds)
        
        with open(failed_text_path, 'w') as failed_txt:
            for video_tensor, fp in tqdm(self.dataloader):
                try:
                    if video_tensor is not None:
                        output_ft_path = self._get_output_path(fp)
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
        
        print(f"\n=== {FEAT_NAME} 特征提取统计 ===")
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
            
            np.save(f"./pretrained_zeros/Loconet_zero_oneframe.npy", video_embed.detach().cpu().numpy())




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
    parser = argparse.ArgumentParser(
        description="Extract LoCoNet visual features",
        epilog='Examples:\n'
               '  python extract_visual_features_LoCoNet.py input_dir output_dir\n'
               '  python extract_visual_features_LoCoNet.py --list train_list.txt\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default=None,
                        help="Input video directory (contains .mp4 files)")
    parser.add_argument("output_dir", nargs='?', default=None,
                        help="Output feature directory (will mirror input structure)")
    parser.add_argument("--list", nargs='+', default=None, dest='file_lists',
                        help="List file(s) with mp4 absolute paths (output: /mp4/ -> /LoCoNet/)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers")
    args = parser.parse_args()
    
    use_list = args.file_lists is not None
    use_dir = args.input_dir is not None and args.output_dir is not None
    if not use_list and not use_dir:
        parser.error("Either provide input_dir output_dir, or --list file(s)")
    
    model_path = os.path.join(config.LOCONET_PATH, "loconet_ava_best.model")
    
    # Download pretrained model if not exists
    if not os.path.isfile(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the model manually:")
        print("1. Visit: https://drive.google.com/file/d/1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK/view")
        print(f"2. Save it to: {model_path}")
        print("\nOr use this command (if you have network access to Google Drive):")
        print(f"   gdown 1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK -O {model_path}")
        
        # Try to download
        print("\nAttempting to download...")
        Link = "1EX-V464jCD6S-wg68yGuAa-UcsMrw8mK"
        cmd = f"gdown {Link} -O {model_path}"
        import subprocess
        result = subprocess.call(cmd, shell=True)
        
        if result != 0 or not os.path.isfile(model_path):
            print(f"\nDownload failed! Please download manually.")
            return
        print(f"Model downloaded successfully to {model_path}")
    
    process = TalkNetBatchedPreprocessing(
        model_path,
        data_path=args.input_dir,
        output_dir=args.output_dir,
        file_list_paths=args.file_lists,
        num_workers=args.num_workers
    )
    process.extract_features()

if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"
    main()
