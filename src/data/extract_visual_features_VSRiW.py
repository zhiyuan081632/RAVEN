
import os
import torch
import numpy as np
import pandas as pd
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sys.path.append(config.VSRIW_PATH)
from pipelines.model import AVSR
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from pathlib import Path
from mediapipe.python.solutions.face_detection import FaceDetection, FaceKeyPoint
from pipelines.detectors.mediapipe.video_process import VideoProcess
from pipelines.data.transforms import VideoTransform

FEAT_NAME = "feats_VSRiW"

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
            face_track: bool = True, 
            ) -> None:
        super().__init__()
        self.data_path = data_path
        if file_list_paths is not None:
            self.file_list = _load_list_files(file_list_paths)
        elif data_path is not None:
            self.file_list = sorted(str(p) for p in Path(data_path).rglob("*.mp4"))
        else:
            raise ValueError("Either data_path or file_list_paths must be provided")
        print(f"Found {len(self.file_list)} video files")
        
        self.video_processor = VideoProcess()
        self.video_transform = VideoTransform(speed_rate=1)
        self.face_track = face_track
        self.short_range_detector: FaceDetection = None
        self.long_range_detector: FaceDetection = None 

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        fp = self.file_list[index]
        video_tensor = torchvision.io.read_video(fp, pts_unit='sec')[0].numpy()
        video_tensor = random_5s_clip(video_tensor)
        try:
            landmarks = self._process_landmarks(video_tensor)
            video_tensor = self.video_processor(video_tensor, landmarks)
            video_tensor = self.video_transform(torch.tensor(video_tensor))
            return video_tensor, fp
        except:
            return None, fp
        
    def _initialize_detector(self):
        self.short_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.long_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def _process_landmarks(self, video_tensor):
        if self.face_track:
            landmarks = self._detect(video_tensor, self.long_range_detector)
            if all(l is None for l in landmarks):
                landmarks = self._detect(video_tensor, self.short_range_detector)
                if all(l is None for l in landmarks):
                    UserWarning("Cannot detect any frames in the video")
            return landmarks
        else:
            return None
    
    def _detect(self, video_frames, detector):
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


def video_dataset_worker_init(worker_id):
    info = get_worker_info()
    info.dataset._initialize_detector()
    
def random_5s_clip(video, frame_rate=25):

    T, H, W, C = video.shape
    target_frames = 5 * frame_rate 

    if T >= target_frames:
        
        return video[:target_frames, :, :, :]
    else:
        pad_frames = target_frames - T
        pad_array = np.zeros((pad_frames, H, W, C), dtype=video.dtype) 
        return np.concatenate((video, pad_array), axis=0)

class AVSRBatchedPreprocessing:
    def __init__(
            self,
            model_path,
            model_conf,
            data_path=None,
            output_dir=None,
            file_list_paths=None,
            batch_size= None,
            num_workers: int = 1,
            face_track: bool = True, 
            device = "cuda"
        ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.list_mode = file_list_paths is not None
        self.ds = VideoDataset(data_path=data_path, file_list_paths=file_list_paths, face_track=face_track)
        self.dataloader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, worker_init_fn=video_dataset_worker_init)
        self.model = AVSR("video", model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
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
                        if not os.path.exists(os.path.dirname(output_ft_path)):
                            os.makedirs(os.path.dirname(output_ft_path))
                        with torch.no_grad():
                            enc_feats: torch.Tensor = self.model.model.encode(video_tensor.to(self.device), True)
                        
                        features = enc_feats.detach().cpu().numpy()
                        np.save(output_ft_path, features)
                        success_count += 1
                    else:
                        print(f"Video {fp} preprocessing failed.")
                        failed_txt.write(fp + '\n')
                        failure_count += 1
                except Exception as e:
                    print(f"Error processing {fp}: {e}")
                    failed_txt.write(fp + f" # {str(e)}\n")
                    failure_count += 1
        
        print(f"\n=== 特征提取统计 ===")
        print(f"总文件数: {total}")
        print(f"新生成: {success_count}")
        print(f"已跳过: {skip_count}")
        print(f"失败: {failure_count}")
    
    def test_zero(self):
        
        with torch.no_grad():

            video_tensor = torch.zeros(1, 1, 88, 88)
            video_embed = self.model.model.encode(video_tensor.to(self.device), True)
                
   
                
            np.save(f"./pretrained_zeros/VSRiW_zero_oneframe.npy", video_embed.detach().cpu().numpy())
            
            print(video_embed.shape)
            print(video_embed)
                

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract VSRiW visual features",
        epilog='Examples:\n'
               '  python extract_visual_features_VSRiW.py input_dir output_dir\n'
               '  python extract_visual_features_VSRiW.py --list train_list.txt\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", nargs='?', default=None,
                        help="Input video directory (contains .mp4 files)")
    parser.add_argument("output_dir", nargs='?', default=None,
                        help="Output feature directory (will mirror input structure)")
    parser.add_argument("--list", nargs='+', default=None, dest='file_lists',
                        help="List file(s) with mp4 absolute paths (output: /mp4/ -> /VSRiW/)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers")
    args = parser.parse_args()
    
    use_list = args.file_lists is not None
    use_dir = args.input_dir is not None and args.output_dir is not None
    if not use_list and not use_dir:
        parser.error("Either provide input_dir output_dir, or --list file(s)")
    
    model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
    model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
    
    process = AVSRBatchedPreprocessing(
        model_path=model_path,
        model_conf=model_conf,
        data_path=args.input_dir,
        output_dir=args.output_dir,
        file_list_paths=args.file_lists,
        face_track=True,
        num_workers=args.num_workers
    )
    process.extract_features()


if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"

    main()
    
    