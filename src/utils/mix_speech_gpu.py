import random
import sys
import os

import soundfile as sf

import hashlib

# Add parent directory to path to enable both direct script execution and module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .utils import crop_pad_audio, apply_gain
except ImportError:
    from utils.utils import crop_pad_audio, apply_gain
    
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import librosa
from concurrent.futures import ProcessPoolExecutor
import config

# Change SPLIT to "train" or "val" or "test" to process the corresponding dataset
SPLIT = "train"
# Input directory - set to either "dev/wav" or "dev/aac"
INPUT_DIR = "dev/aac"  # Change this to "dev/wav" if you have wav files
OUTPUT_DIR = os.path.join(config.DATA_FOLDER_PATH, "dev/mixed_wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)

voxceleb2_split_fp = os.path.join(config.PROJECT_ROOT, "src/data/split.parquet")
voxceleb2_fps = pd.read_parquet(voxceleb2_split_fp)
voxceleb2_fps = voxceleb2_fps[voxceleb2_fps["split"] == SPLIT]["audio_fp"]
musan_split_fp = os.path.join(config.PROJECT_ROOT, "src/data/musan_split.csv")
musan_fps = pd.read_csv(musan_split_fp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_stream = torch.cuda.Stream()  


def augment_speech(target_speaker, target_speaker_fp, voxceleb2_fps, musan_fps, snr_lower, snr_upper, split):
    """ Perform augmentation with GPU acceleration. """
    sample_seed = hash(target_speaker_fp) % (2**32)
    torch.manual_seed(sample_seed)
    
    augment_type = np.random.choice(["music", "noise", "speech", "none"])

    if augment_type in ["music", "noise"]:
        musan_fps = musan_fps[(musan_fps["type"] == augment_type) & (musan_fps["split"] == split)]["filepath"]
    else:
        musan_fps = None

    interfering_speaker_fp, other_interference_fp = get_interfering_fps(target_speaker_fp, musan_fps, voxceleb2_fps, augment_type)

    mixed_audio = mix(target_speaker, interfering_speaker_fp, other_interference_fp, snr_lower, snr_upper)
    return mixed_audio


def get_interfering_fps(target_speaker_fp, musan_fps, voxceleb2_fps, augment_type):
    voxceleb2_fps = voxceleb2_fps[voxceleb2_fps != target_speaker_fp]
    interfering_speaker_fp = np.random.choice(voxceleb2_fps)
    interfering_speaker_fp = os.path.join(config.DATA_FOLDER_PATH, interfering_speaker_fp)

    if augment_type == "none":
        other_interference_fp = None
    elif augment_type == "speech":
        voxceleb2_fps = voxceleb2_fps[voxceleb2_fps != interfering_speaker_fp]
        other_interference_fp = np.random.choice(voxceleb2_fps)
        other_interference_fp = os.path.join(config.DATA_FOLDER_PATH, other_interference_fp)
    else:
        other_interference_fp = np.random.choice(musan_fps)
        other_interference_fp = os.path.join(config.MUSAN_FOLDER_PATH, other_interference_fp)

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


def process_file(file_info):
    """ Process a single file in parallel, skipping already processed files. """
    m4a_path, input_dir = file_info
    relative_path = m4a_path.relative_to(input_dir)
    output_path = Path(OUTPUT_DIR) / relative_path.parent / (relative_path.stem + ".wav")
    
    if output_path.exists():
        return
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    target_speaker_fp = os.path.join(config.DATA_FOLDER_PATH, m4a_path)
    target_speaker = crop_pad_audio(target_speaker_fp, 16000, 5)

    mixed_audio = augment_speech(target_speaker, m4a_path, voxceleb2_fps, musan_fps, config.TRAINING_LOWER_SNR, config.TRAINING_UPPER_SNR, SPLIT)
    sf.write(output_path, mixed_audio, 16000, format="wav")



def main():
    """ Run multiprocessing with multiple workers, skipping existing files. """
    num_workers = min(os.cpu_count(), 16)
    
    # Detect format from INPUT_DIR
    if "wav" in INPUT_DIR:
        input_format = "wav"
        # Convert aac paths to wav paths
        audio_fps = voxceleb2_fps.str.replace("/aac/", "/wav/").str.replace(".m4a", ".wav")
        print(f"Using WAV format from {INPUT_DIR}")
    else:  # aac
        input_format = "aac"
        # Use original aac paths
        audio_fps = voxceleb2_fps
        print(f"Using AAC/M4A format from {INPUT_DIR}")
    
    audio_files = [Path(fp) for fp in audio_fps.tolist()]
    file_info_list = [(fp, INPUT_DIR) for fp in audio_files]
    
    print(f"Total files to process: {len(audio_files)}")
    print(f"Output directory: {OUTPUT_DIR}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_file, file_info_list), total=len(file_info_list)))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True) 
    os.environ["GLOG_minloglevel"] = "3"
    main()
