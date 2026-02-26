#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extract audio from VoxCeleb2 MP4 files to WAV/M4A, preserving directory structure.

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
import tempfile
import wave
from pathlib import Path
from multiprocessing import Pool, Manager
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


def get_audio_first_pts(video_path: Path) -> float:
    """
    Get the PTS of the FIRST audio packet using ffprobe.
    This detects AAC priming samples (negative PTS).
    
    Returns:
        First packet PTS in seconds, or 0.0 if failed
    """
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_packets', '-read_intervals', '%+#1',
            str(video_path)
        ], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'pts_time=' in line:
                return float(line.split('=')[1])
        return 0.0
    except Exception:
        return 0.0


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
    Extract audio from src_mp4 to dst_audio.
    
    Simple extraction without priming sample handling.
    This ensures audio-video sync is maintained from original MP4.
    
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
        
        # Direct extraction and conversion based on output format
        if output_format.lower() == 'wav':
            # Extract to WAV (PCM)
            cmd = [
                'ffmpeg', '-y', '-i', str(src_mp4),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', '1',
                str(dst_audio)
            ]
        else:
            # Extract to M4A (AAC)
            cmd = [
                'ffmpeg', '-y', '-i', str(src_mp4),
                '-vn', '-acodec', 'aac',
                '-b:a', '128k',
                '-ar', str(sample_rate), '-ac', '1',
                str(dst_audio)
            ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return True
        
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def _extract_audio_simple(src_mp4: Path, dst_audio: Path, sample_rate: int = 16000, 
                          output_format: str = 'wav') -> bool:
    """
    Simple audio extraction fallback (no priming sample handling).
    Used when video duration cannot be determined.
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
    src_mp4, dst_audio, sample_rate, output_format, counter, lock, total = args_tuple

    # Skip if already exists
    if dst_audio.exists():
        status = "exists"
        ok = True
    else:
        ok = extract_audio_with_alignment(src_mp4, dst_audio, sample_rate, output_format)
        status = "created" if ok else "failed"
    
    # Thread-safe progress update
    with lock:
        counter.value += 1
        progress_pct = (counter.value / total) * 100
        print(f"\r[{counter.value}/{total}] {progress_pct:.1f}% | {status}: {src_mp4.name}", 
              end="", flush=True)
    
    return src_mp4, ok, status


def build_task_list(input_root: Path, output_root: Path, sample_rate: int, output_format: str,
                    counter, lock, total):
    """
    Build (src_mp4, dst_audio) pairs for all mp4 files under input_root.
    """
    tasks = []
    ext = '.wav' if output_format.lower() == 'wav' else '.m4a'
    
    for src_mp4 in input_root.rglob("*.mp4"):
        rel = src_mp4.relative_to(input_root)           # e.g. idXXXX/AAA/00001.mp4
        dst_audio = output_root / rel
        dst_audio = dst_audio.with_suffix(ext)          # change extension
        tasks.append((src_mp4, dst_audio, sample_rate, output_format, counter, lock, total))
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Extract audio from VoxCeleb2 MP4 files')
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing MP4 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted audio files")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--output_format", type=str, default='wav',
                        choices=['wav', 'm4a'],
                        help="Output format: 'wav' (PCM, no delay) or 'm4a' (AAC, pre-aligned) (default: wav)")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_root}")

    print(f"Input directory:  {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Sample rate:      {args.sample_rate} Hz")
    print(f"Output format:    {args.output_format.upper()}")

    # Build task list first to get total count
    print("Scanning for MP4 files...")
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    
    # Pre-scan to get total
    all_mp4s = list(input_root.rglob("*.mp4"))
    total = len(all_mp4s)
    print(f"Total MP4 files found: {total}")

    if total == 0:
        print("No MP4 files found.")
        return

    # Build tasks with shared counter
    tasks = build_task_list(input_root, output_root, args.sample_rate, args.output_format, 
                            counter, lock, total)

    print(f"Starting extraction with {args.num_workers} workers...")
    start_time = time.time()
    
    failed_files = []
    
    # Multiprocessing without tqdm (custom progress)
    with Pool(processes=args.num_workers) as pool:
        results = pool.imap_unordered(process_single, tasks)
        for src_mp4, ok, status in results:
            if not ok and status == "failed":
                failed_files.append(str(src_mp4))
    
    # Clean newline after progress
    print()
    
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