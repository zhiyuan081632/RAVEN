import pytorch_lightning as pl
from torch.utils import data
from data.dataset import VoxCeleb2
import torch
import glob
import os


class DataModule(pl.LightningDataModule):

    def __init__(self, visual_encoder, embedding_size, batch_size=4, num_workers=2,
                 speech_train_lists=None, speech_val_lists=None, speech_test_lists=None,
                 noise_lists=None, music_lists=None):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.visual_encoder = visual_encoder
        self.embedding_size = embedding_size
        # speech list 文件
        self.speech_train_lists = speech_train_lists or []
        self.speech_val_lists = speech_val_lists or []
        self.speech_test_lists = speech_test_lists or []
        # noise/music list 文件（按 split 分组）
        self.noise_lists = noise_lists or {}
        self.music_lists = music_lists or {}

    def setup(self, test_condition=None, test_snr=None):
        self.train_dataset = VoxCeleb2(
            "train", self.visual_encoder, self.embedding_size,
            speech_lists=self.speech_train_lists,
            noise_lists=self.noise_lists.get("train"),
            music_lists=self.music_lists.get("train"),
        )
        self.val_dataset = VoxCeleb2(
            "val", self.visual_encoder, self.embedding_size,
            speech_lists=self.speech_val_lists,
            noise_lists=self.noise_lists.get("val"),
            music_lists=self.music_lists.get("val"),
        )
        self.test_dataset = VoxCeleb2(
            "test", self.visual_encoder, self.embedding_size,
            speech_lists=self.speech_test_lists,
            noise_lists=self.noise_lists.get("test"),
            music_lists=self.music_lists.get("test"),
            condition=test_condition, snr=test_snr,
        )


    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self, small=True):
        if small:
            subset_sampler = data.SubsetRandomSampler(range(1000))
            return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, sampler=subset_sampler, persistent_workers=True, prefetch_factor=2)
        else:
            return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn_custom, persistent_workers=True, prefetch_factor=2)
    


    def collate_fn_custom(self, batch):
        # 过滤缺失数据（__getitem__ 返回 None 的条目）
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return {
            'face_embed': torch.stack([x['face_embed'] for x in batch]).to("cuda", non_blocking=True), 
            'input_audio': torch.stack([x['input_audio'] for x in batch]).to("cuda", non_blocking=True),
            'mixed_audio': torch.stack([x['mixed_audio'] for x in batch]).to("cuda", non_blocking=True),
            'audio_fp': [x['audio_fp'] for x in batch],
            'interfering_speaker_fp': [x['interfering_speaker_fp'] for x in batch],
        }


if __name__ == "__main__":
    
    import time
    import cProfile
    datamodule = VoxCeleb2DataModule(visual_encoder="TalkNet" , embedding_size=1024 ,batch_size=4, num_workers=4)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i == 10:
            break
    print(f"Time to load 1 batches: {time.time() - start_time} seconds")