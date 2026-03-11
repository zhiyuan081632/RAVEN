#!/usr/bin/env python
"""
Generate split.csv for ChineseLips dataset based on mp4 and wav directories.
Format compatible with split.csv: audio_fp,video_fp,split,dataset
"""

import os
import pandas as pd

DATA_ROOT = "/mnt/e/data/ChineseLips"

def scan_split(split_name):
    """Scan mp4/wav directories for a given split (train/val/test)."""
    mp4_dir = os.path.join(DATA_ROOT, split_name, "mp4")
    wav_dir = os.path.join(DATA_ROOT, split_name, "wav")
    
    if not os.path.exists(mp4_dir):
        print(f"Warning: {mp4_dir} does not exist")
        return []
    
    mp4_files = sorted(os.listdir(mp4_dir))
    
    records = []
    for mp4_file in mp4_files:
        if not mp4_file.endswith(".mp4"):
            continue
        
        # Corresponding wav file
        wav_file = mp4_file.replace(".mp4", ".wav")
        
        audio_fp = os.path.join(split_name, "wav", wav_file)
        video_fp = os.path.join(split_name, "mp4", mp4_file)
        
        records.append({
            "audio_fp": audio_fp,
            "video_fp": video_fp,
            "split": split_name,
            "dataset": "ChineseLips"
        })
    
    return records

def main():
    all_records = []
    
    for split_name in ["train", "val", "test"]:
        records = scan_split(split_name)
        all_records.extend(records)
        print(f"{split_name}: {len(records)} files")
    
    df = pd.DataFrame(all_records)
    
    # Save to CSV
    output_path = os.path.join(DATA_ROOT, "split_chineselips.csv")
    df.to_csv(output_path, index=False)
    print(f"\nTotal: {len(df)} records")
    print(f"Saved to: {output_path}")
    
    # Show sample
    print("\nSample:")
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
