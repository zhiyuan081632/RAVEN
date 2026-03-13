import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.utils import crop_pad_audio
import librosa
import hashlib
import soundfile as sf
from tqdm import tqdm
import config
from data.dataset import _load_list_files, _mp4_to_wav




def generate_noise_condition(target_fp, noise_fps, snr, orig_sr=16000, crop_length=5):
    
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    rng = np.random.RandomState(seed)
    noise_fp = rng.choice(noise_fps)
    noise_audio = crop_pad_audio(noise_fp, orig_sr, crop_length)
    noise_audio = librosa.util.normalize(noise_audio)
    
    mixed_audio = mix_per_snr(target_audio, noise_audio, snr)
    
    return mixed_audio


def generate_one_interfering_speaker_condition(target_fp, all_wav_fps, snr, orig_sr=16000, crop_length=5):
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    rng = np.random.RandomState(seed)
    candidates = [fp for fp in all_wav_fps if fp != target_fp]
    interfering_speaker_fp = rng.choice(candidates)
    interfering_speaker_audio = crop_pad_audio(interfering_speaker_fp, orig_sr, crop_length)
    interfering_speaker_audio = librosa.util.normalize(interfering_speaker_audio)
    
    mixed_audio = mix_per_snr(target_audio, interfering_speaker_audio, snr)
    
    return mixed_audio

def generate_three_interfering_speakers_condition(target_fp, all_wav_fps, snr, orig_sr=16000, crop_length=5):
    target_audio = crop_pad_audio(target_fp, orig_sr, crop_length)
    target_audio = librosa.util.normalize(target_audio)
    seed = int(hashlib.md5(target_fp.encode()).hexdigest(), 16) % (10 ** 8)
    rng = np.random.RandomState(seed)
    candidates = [fp for fp in all_wav_fps if fp != target_fp]
    interfering_fps = rng.choice(candidates, size=3, replace=False)
    interfering_speaker_audio = [crop_pad_audio(fp, orig_sr, crop_length) for fp in interfering_fps]
    interfering_speaker_audio = [librosa.util.normalize(audio) for audio in interfering_speaker_audio]
    interfering_speaker_audio = np.sum(interfering_speaker_audio, axis=0)
    assert len(interfering_speaker_audio) == len(target_audio)
    
    mixed_audio = mix_per_snr(target_audio, interfering_speaker_audio, snr)
    
    return mixed_audio
    




def main(condition, snr):
    # 从 config 读取测试集 list
    mp4_fps = _load_list_files(config.SPEECH_TEST_LISTS)
    all_wav_fps = [_mp4_to_wav(fp) for fp in mp4_fps]
    
    # 噪声
    noise_fps = _load_list_files(config.NOISE_LISTS.get("test", []))
    
    print(f"Test speech files: {len(all_wav_fps)}")
    print(f"Test noise files: {len(noise_fps)}")
    
    for i, wav_fp in tqdm(enumerate(all_wav_fps), total=len(all_wav_fps)):
        if condition == "noise_only":
            mixed_audio = generate_noise_condition(wav_fp, noise_fps, snr)
        elif condition == "one_interfering_speaker":
            mixed_audio = generate_one_interfering_speaker_condition(wav_fp, all_wav_fps, snr)
        elif condition == "three_interfering_speakers":
            mixed_audio = generate_three_interfering_speakers_condition(wav_fp, all_wav_fps, snr)
        
        # 输出路径: /wav/ -> /mixed_wav/{condition}/{snr}/
        mixed_audio_fp = wav_fp.replace('/wav/', f'/mixed_wav/{condition}/{snr}/')
        os.makedirs(os.path.dirname(mixed_audio_fp), exist_ok=True)
        
        sf.write(mixed_audio_fp, mixed_audio, 16000)
        

def mix_per_snr(target_speaker, all_interference, dB_snr):
    """ Mix target speaker with interfering speaker and noise on GPU. """

    if dB_snr == "mixed":
        dB_snr = np.random.uniform(-10, 10)
    else:
        dB_snr = int(dB_snr)
    snr_factor = 10**(dB_snr / 10)
    
    target_speaker_power = np.mean(target_speaker**2)
    all_interference_power = np.mean(all_interference**2)

    scale_factor = np.sqrt(target_speaker_power / (snr_factor * all_interference_power))
    mixed_audio = target_speaker + all_interference * scale_factor
    mixed_audio = librosa.util.normalize(mixed_audio)
    
    return mixed_audio
    
    
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data under various conditions.")
    parser.add_argument("--condition", type=str, required=True,
                        help='Condition(s), comma-separated if multiple: noise_only,one_interfering_speaker')
    parser.add_argument("--snr", type=str, required=True,
                        help='SNR value(s), comma-separated if multiple: mixed,-10,-5,0')

    args = parser.parse_args()

    conditions = [c.strip() for c in args.condition.split(",")]
    snrs = [s.strip() for s in args.snr.split(",")]

    for condition in conditions:
        for snr in snrs:
            print(f"Running condition: {condition}, SNR: {snr}")
            main(condition, snr)
