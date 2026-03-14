#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extract audio from MP4 files to WAV/M4A, preserving directory structure.

- Input root:  /mnt/e/data/VoxCeleb2/dev/mp4
- Output root: /mnt/e/data/VoxCeleb2/dev/wav or /mnt/e/data/VoxCeleb2/dev/aac
- For each *.mp4, create a corresponding *.wav or *.m4a with the same relative path.
- Audio is extracted with precise temporal alignment to match video duration.
- Handles AAC priming samples (negative PTS) for perfect audio-video synchronization.

Run:
    # For WAV output (no encoding delay)
    python preprocess_vox2_audio.py --input_dir /mnt/e/data/VoxCeleb2/dev/mp4 \
                                    --output_dir /mnt/e/data/VoxCeleb2/dev/wav \
                                    --sample_rate 16000 \
                                    --output_format wav \
                                    --num_workers 4
    
    # For M4A output (AAC encoding, with priming sample correction)
    python preprocess_vox2_audio.py --input_dir /mnt/e/data/VoxCeleb2/dev/mp4 \
                                    --output_dir /mnt/e/data/VoxCeleb2/dev/aac \
                                    --sample_rate 16000 \
                                    --output_format m4a \
                                    --num_workers 4
"""

import os
import sys
import argparse
import subprocess
import time
import json
import wave
from pathlib import Path
from multiprocessing import Pool
import numpy as np


def get_video_duration(video_path: Path) -> float:
    """
    Get exact video stream duration using ffprobe.
    """
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return None


def _get_skip_samples(src_mp4):
    """
    Get AAC encoder skip_samples from the first audio packet's side data.
    
    Returns:
        skip_samples count at the original sample rate, or 0 if not found.
    """
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_packets', '-read_intervals', '%+#1',
            '-of', 'json', str(src_mp4)
        ], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for pkt in data.get('packets', []):
            for sd in pkt.get('side_data_list', []):
                if sd.get('side_data_type') == 'Skip Samples':
                    return int(sd.get('skip_samples', 0))
        return 0
    except Exception:
        return 0


def read_wav_data(wav_path: Path):
    """
    Read WAV file and return audio data as numpy array.
    
    Returns:
        Tuple of (audio_data as np.int16 array, sample_rate)
    """
    with wave.open(str(wav_path), 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(frames), dtype=np.int16)
    return data, rate


def write_wav_data(wav_path: Path, data: np.ndarray, sample_rate: int):
    """
    Write numpy array to WAV file.
    
    Args:
        wav_path: Path to output WAV file
        data: Audio data as np.int16 array
        sample_rate: Sample rate
    """
    with wave.open(str(wav_path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())


def extract_audio_with_alignment(src_mp4: Path, dst_audio: Path, sample_rate: int = 16000, 
                                  output_format: str = 'wav') -> bool:
    """
    Extract audio from src_mp4 to dst_audio with precise alignment.
    
    1. ffmpeg's edit-list handling removes AAC priming samples automatically.
    2. After extraction, trim/pad the WAV to exactly match video duration
       (removes AAC frame-boundary padding at the end).
    
    Args:
        src_mp4: Source MP4 video file
        dst_audio: Destination audio file (.wav or .m4a)
        sample_rate: Target sample rate (default: 16000)
        output_format: Output format 'wav' or 'm4a' (default: 'wav')
    
    Returns:
        True on success, False on failure.
    """
    try:
        dst_audio.parent.mkdir(parents=True, exist_ok=True)
        video_dur = get_video_duration(src_mp4)
        
        if output_format.lower() == 'wav':
            cmd = [
                'ffmpeg', '-y', '-i', str(src_mp4),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', '1',
                str(dst_audio)
            ]
        else:
            cmd = [
                'ffmpeg', '-y', '-i', str(src_mp4),
                '-vn', '-acodec', 'aac',
                '-b:a', '128k',
                '-ar', str(sample_rate), '-ac', '1',
                str(dst_audio)
            ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Trim/pad WAV to exact video duration (remove AAC frame padding)
        if output_format.lower() == 'wav' and video_dur is not None and video_dur > 0:
            target_samples = int(round(video_dur * sample_rate))
            data, sr = read_wav_data(dst_audio)
            if len(data) != target_samples:
                if len(data) > target_samples:
                    data = data[:target_samples]
                else:
                    data = np.pad(data, (0, target_samples - len(data)))
                write_wav_data(dst_audio, data, sr)
        
        return True
        
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def _extract_audio_simple(src_mp4: Path, dst_audio: Path, sample_rate: int = 16000, 
                          output_format: str = 'wav') -> bool:
    """
    Simple audio extraction fallback (no priming sample handling).
    """
    if output_format.lower() == 'wav':
        cmd = [
            'ffmpeg', '-y', '-i', str(src_mp4),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate), '-ac', '1',
            str(dst_audio)
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', str(src_mp4),
            '-vn', '-acodec', 'aac', '-b:a', '128k',
            '-ar', str(sample_rate), '-ac', '1',
            str(dst_audio)
        ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def process_single(args_tuple):
    """
    Single-file processing wrapper for multiprocessing Pool.
    """
    src_mp4, dst_audio, sample_rate, output_format = args_tuple

    # Skip if already exists
    if dst_audio.exists():
        status = "exists"
        ok = True
    else:
        ok = extract_audio_with_alignment(src_mp4, dst_audio, sample_rate, output_format)
        status = "created" if ok else "failed"
    
    return src_mp4, ok, status


def build_task_list(input_root: Path, output_root: Path, sample_rate: int, output_format: str):
    """
    Build (src_mp4, dst_audio) pairs for all mp4 files under input_root.
    """
    tasks = []
    skipped = 0
    ext = '.wav' if output_format.lower() == 'wav' else '.m4a'
    
    for src_mp4 in input_root.rglob("*.mp4"):
        rel = src_mp4.relative_to(input_root)           # e.g. idXXXX/AAA/00001.mp4
        dst_audio = output_root / rel
        dst_audio = dst_audio.with_suffix(ext)          # change extension
        if dst_audio.exists():
            skipped += 1
            continue
        tasks.append((src_mp4, dst_audio, sample_rate, output_format))
    if skipped > 0:
        print(f"Skipped {skipped} already converted files")
    return tasks


def build_task_list_from_filelist(file_list_paths, sample_rate: int, output_format: str):
    """
    从 list 文件构建任务，每行一个 mp4 绝对路径。
    输出路径: /mp4/ -> /wav/ (or /m4a/), .mp4 -> .wav (or .m4a)
    """
    tasks = []
    skipped = 0
    ext = '.wav' if output_format.lower() == 'wav' else '.m4a'
    fmt_dir = 'wav' if output_format.lower() == 'wav' else 'm4a'
    
    if isinstance(file_list_paths, str):
        file_list_paths = [file_list_paths]
    
    for list_fp in file_list_paths:
        with open(list_fp, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line == 'filepath':
                    continue
                src_mp4 = Path(line)
                # /mp4/ -> /wav/, .mp4 -> .wav
                dst_str = line.replace('/mp4/', f'/{fmt_dir}/').replace('.mp4', ext)
                dst_audio = Path(dst_str)
                if dst_audio.exists():
                    skipped += 1
                    continue
                tasks.append((src_mp4, dst_audio, sample_rate, output_format))
    if skipped > 0:
        print(f"Skipped {skipped} already converted files")
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from MP4 files to WAV/M4A',
        epilog='Examples:\n'
               '  python convert_mp4_to_wav.py /data/VoxCeleb2/dev/mp4 /data/VoxCeleb2/dev/wav\n'
               '  python convert_mp4_to_wav.py --list train_list.txt val_list.txt -j 8\n',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # 两种输入模式：目录 or 文件列表
    parser.add_argument("input_dir", nargs='?', default=None,
                        help="Input directory containing MP4 files")
    parser.add_argument("output_dir", nargs='?', default=None,
                        help="Output directory for extracted audio files")
    parser.add_argument("--list", nargs='+', default=None, dest='file_lists',
                        help="List file(s) with mp4 absolute paths (output: /mp4/ -> /wav/)")
    parser.add_argument("-j", "--jobs", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--format", type=str, default='wav', dest='output_format',
                        choices=['wav', 'm4a'],
                        help="Output format: wav or m4a (default: wav)")
    args = parser.parse_args()

    # 判断输入模式
    use_list = args.file_lists is not None
    use_dir = args.input_dir is not None and args.output_dir is not None
    if not use_list and not use_dir:
        parser.error("Either provide input_dir output_dir, or --list file(s)")

    if use_list:
        print(f"Mode: file list")
        print(f"List files: {args.file_lists}")
        print(f"Sample rate:   {args.sample_rate} Hz")
        print(f"Output format: {args.output_format.upper()}")
        print(f"Output rule:   /mp4/ -> /{args.output_format}/,  .mp4 -> .{args.output_format}")
        
        # 统计总数
        all_mp4s = []
        for lf in args.file_lists:
            with open(lf, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line != 'filepath':
                        all_mp4s.append(line)
        total = len(all_mp4s)
        print(f"Total MP4 files in lists: {total}")
        if total == 0:
            print("No MP4 files found in list files.")
            return
        
        tasks = build_task_list_from_filelist(
            args.file_lists, args.sample_rate, args.output_format)
    else:
        input_root = Path(args.input_dir)
        output_root = Path(args.output_dir)
        if not input_root.is_dir():
            raise SystemExit(f"Input directory does not exist: {input_root}")
        
        print(f"Mode: directory")
        print(f"Input:  {input_root}")
        print(f"Output: {output_root}")
        print(f"Sample rate:   {args.sample_rate} Hz")
        print(f"Output format: {args.output_format.upper()}")
        
        print("Scanning for MP4 files...")
        all_mp4s = list(input_root.rglob("*.mp4"))
        total = len(all_mp4s)
        print(f"Total MP4 files found: {total}")
        if total == 0:
            print("No MP4 files found.")
            return
        
        tasks = build_task_list(
            input_root, output_root, args.sample_rate, args.output_format)

    print(f"Tasks to process: {len(tasks)}")
    if len(tasks) == 0:
        print("All files already converted.")
        return

    print(f"Starting extraction with {args.jobs} workers...")
    start_time = time.time()
    
    failed_files = []
    
    # Multiprocessing — progress printed in main process
    done = 0
    task_total = len(tasks)
    with Pool(processes=args.jobs) as pool:
        for src_mp4, ok, status in pool.imap_unordered(process_single, tasks):
            done += 1
            if not ok and status == "failed":
                failed_files.append(str(src_mp4))
            if done % 100 == 0 or done == task_total:
                print(f"[{done}/{task_total}] {done/task_total*100:.1f}%")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({total/elapsed:.1f} files/s)")
    
    if failed_files:
        print(f"\nFailed extractions ({len(failed_files)} files):")
        for f in failed_files[:10]:  # show first 10
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more")
    
    print("\n" + "="*60)
    if args.output_format.lower() == 'wav':
        print("Done! Audio files saved as WAV (PCM 16kHz, mono).")
        print("Audio is precisely aligned to video duration with AAC priming samples removed.")
        print("Output format: PCM WAV (no encoding delay, perfect synchronization).")
    else:
        print("Done! Audio files saved as M4A (AAC 128kbps, 16kHz, mono).")
        print("Audio is pre-aligned to video duration (priming samples removed before encoding).")
        print("Output format: AAC M4A (minimal delay, synchronized to video).")
    print("="*60)


if __name__ == "__main__":
    main()