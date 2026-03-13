import random
import sys
import os
import argparse

import soundfile as sf

# Add parent directory to path to enable both direct script execution and module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utils import crop_pad_audio, apply_gain
except ImportError:
    from utils.utils import crop_pad_audio, apply_gain

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
import config
from data.dataset import _load_list_files, _mp4_to_wav


def augment_speech(target_speaker, target_wav_fp, all_wav_fps, noise_fps, music_fps, snr_lower, snr_upper):
    """ Perform augmentation. """
    sample_seed = hash(target_wav_fp) % (2**32)
    torch.manual_seed(sample_seed)
    
    augment_type = np.random.choice(["music", "noise", "speech", "none"])

    interfering_speaker_fp, other_interference_fp = get_interfering_fps(
        target_wav_fp, all_wav_fps, noise_fps, music_fps, augment_type)

    mixed_audio = mix(target_speaker, interfering_speaker_fp, other_interference_fp, snr_lower, snr_upper)
    return mixed_audio


def get_interfering_fps(target_wav_fp, all_wav_fps, noise_fps, music_fps, augment_type):
    # 随机选一个干扰语音（排除自身）
    candidates = [fp for fp in all_wav_fps if fp != target_wav_fp]
    interfering_speaker_fp = np.random.choice(candidates)

    if augment_type == "none":
        other_interference_fp = None
    elif augment_type == "speech":
        candidates2 = [fp for fp in candidates if fp != interfering_speaker_fp]
        other_interference_fp = np.random.choice(candidates2)
    elif augment_type == "noise" and noise_fps:
        other_interference_fp = np.random.choice(noise_fps)
    elif augment_type == "music" and music_fps:
        other_interference_fp = np.random.choice(music_fps)
    else:
        other_interference_fp = None

    return interfering_speaker_fp, other_interference_fp


def mix(target_speaker, interfering_speaker_fp, other_interference_fp, snr_lower, snr_upper, orig_sr=16000, crop_length=5):
    """ Mix target speaker with interfering speaker and noise. """
    # Process on CPU to avoid GPU memory issues in multiprocessing
    interfering_speaker = crop_pad_audio(interfering_speaker_fp, orig_sr, crop_length)
    interfering_speaker = librosa.util.normalize(interfering_speaker)
    
    if other_interference_fp is not None:
        other_interference = crop_pad_audio(other_interference_fp, orig_sr, crop_length)
        other_interference = librosa.util.normalize(other_interference)
        mixed_interference = other_interference + interfering_speaker
    else:
        mixed_interference = interfering_speaker

    dB_snr = np.random.uniform(snr_lower, snr_upper)
    snr_factor = 10**(dB_snr / 10)
    
    target_speaker_power = np.mean(target_speaker**2)
    mixed_interference_power = np.mean(mixed_interference**2)

    scale_factor = np.sqrt(target_speaker_power / (snr_factor * mixed_interference_power))
    mixed_audio = target_speaker + mixed_interference * scale_factor

    return mixed_audio


# 全局变量，由 _init_worker 初始化，供 process_file 使用
ALL_WAV_FPS = []
NOISE_FPS = []
MUSIC_FPS = []


def _init_worker(all_wav, noise, music):
    """子进程初始化，设置全局变量"""
    global ALL_WAV_FPS, NOISE_FPS, MUSIC_FPS
    ALL_WAV_FPS = all_wav
    NOISE_FPS = noise
    MUSIC_FPS = music


def process_file(wav_fp):
    """ Process a single file, skipping already processed files. """
    # 输出路径: /wav/ -> /mixed_wav/
    output_path = wav_fp.replace('/wav/', '/mixed_wav/')
    
    if os.path.exists(output_path):
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    target_speaker = crop_pad_audio(wav_fp, 16000, 5)
    target_speaker = librosa.util.normalize(target_speaker)

    mixed_audio = augment_speech(
        target_speaker, wav_fp, ALL_WAV_FPS, NOISE_FPS, MUSIC_FPS,
        config.TRAINING_LOWER_SNR, config.TRAINING_UPPER_SNR)
    sf.write(output_path, mixed_audio, 16000, format="wav")


def main(speech_list, noise_list_files, music_list_files):
    """ Run multiprocessing with multiple workers, skipping existing files. """
    global ALL_WAV_FPS, NOISE_FPS, MUSIC_FPS
    
    num_workers = min(os.cpu_count(), 16)
    
    # 加载 speech list（mp4 绝对路径）并转为 wav 路径
    mp4_fps = _load_list_files(speech_list)
    ALL_WAV_FPS = [_mp4_to_wav(fp) for fp in mp4_fps]
    
    # 加载噪声/音乐
    NOISE_FPS = _load_list_files(noise_list_files) if noise_list_files else []
    MUSIC_FPS = _load_list_files(music_list_files) if music_list_files else []
    
    print(f"Speech files: {len(ALL_WAV_FPS)}")
    print(f"Noise files: {len(NOISE_FPS)}")
    print(f"Music files: {len(MUSIC_FPS)}")
    print(f"Sample wav: {ALL_WAV_FPS[0] if ALL_WAV_FPS else 'N/A'}")
    print(f"Sample output: {ALL_WAV_FPS[0].replace('/wav/', '/mixed_wav/') if ALL_WAV_FPS else 'N/A'}")

    with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(ALL_WAV_FPS, NOISE_FPS, MUSIC_FPS)
    ) as executor:
        list(tqdm(executor.map(process_file, ALL_WAV_FPS), total=len(ALL_WAV_FPS)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix speech with noise for training/validation")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Which split to process (default: train)")
    args = parser.parse_args()
    
    # 从 config 中读取对应 split 的 list 文件
    split_speech_map = {
        "train": config.SPEECH_TRAIN_LISTS,
        "val":   config.SPEECH_VAL_LISTS,
        "test":  config.SPEECH_TEST_LISTS,
    }
    speech_lists = split_speech_map[args.split]
    noise_lists = config.NOISE_LISTS.get(args.split, [])
    music_lists = config.MUSIC_LISTS.get(args.split, [])
    
    print(f"Split: {args.split}")
    print(f"Speech lists: {speech_lists}")
    print(f"Noise lists: {noise_lists}")
    print(f"Music lists: {music_lists}")
    
    torch.multiprocessing.set_start_method('spawn', force=True) 
    os.environ["GLOG_minloglevel"] = "3"
    main(speech_lists, noise_lists, music_lists)
