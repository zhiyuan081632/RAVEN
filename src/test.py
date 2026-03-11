import argparse
from models.system import System
from models.fusion import Fusion
from data.datamodule import VoxCeleb2DataModule
from losses.complex_mse import PSA_MSE
from pytorch_lightning import Trainer
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
import torch.nn as nn
import os
import multiprocessing
import warnings
import config


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


def test(speech_dataset, visual_encoder, ckpt_path, test_condition, test_snr, embedding_size, batch_size, num_workers):
    # 根据参数选择数据集路径
    speech_folder_path = config.SPEECH_DATASETS.get(speech_dataset, config.SPEECH_FOLDER_PATH)
    print(f"Using dataset: {speech_dataset} -> {speech_folder_path}")
    
    datamodule = VoxCeleb2DataModule(
        data_path=speech_folder_path,
        visual_encoder=visual_encoder,
        embedding_size=embedding_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    datamodule.setup(test_condition=test_condition, test_snr=test_snr)
    test_loader = datamodule.test_dataloader()

    fusion_network = Fusion(embedding_size=embedding_size)
    loss = PSA_MSE()
    metrics = {
        'snr': SignalNoiseRatio(),
        'pesq': PerceptualEvaluationSpeechQuality(16000, 'wb'),
        'sisdr': ScaleInvariantSignalDistortionRatio(),
        'estoi': ShortTimeObjectiveIntelligibility(16000, True)
    }

    test_meta = {
        "visual_encoder": visual_encoder,
        "ckpt_path": ckpt_path,
        "test_condition": test_condition,
        "test_snr": test_snr,
        "log_path": "test_results.log",
    }

    model = System(fusion_network, loss, metrics, test_meta=test_meta)
    trainer = Trainer()
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AV fusion model")

    parser.add_argument("--speech_dataset", type=str, default=config.DEFAULT_SPEECH_DATASET,
                        choices=list(config.SPEECH_DATASETS.keys()),
                        help="Speech dataset to use")
    parser.add_argument("--visual_encoder", type=str, default=config.VISUAL_ENCODER,
                        choices=config.TEST_VISUAL_ENCODERS,
                        help="Visual encoder to use")
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--test_condition", type=str, default=config.TEST_CONDITION,
                        choices=config.TEST_ALL_CONDITIONS,
                        help="Test condition")
    parser.add_argument("--test_snr", type=str, default=str(config.TEST_SNR),
                        choices=[str(s) for s in config.TEST_ALL_SNRs],
                        help="Test SNR value")
    parser.add_argument("--embedding_size", type=int,
                        help="Embedding size; defaults to encoder-specific value")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader")

    args = parser.parse_args()

    embedding_size = args.embedding_size or config.embedding_size_dict[args.visual_encoder]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    os.environ["GLOG_minloglevel"] = "3"
    multiprocessing.set_start_method('spawn', True)
    
    # 根据数据集获取实际路径
    speech_folder_path = config.SPEECH_DATASETS.get(args.speech_dataset, config.SPEECH_FOLDER_PATH)
    
    print("speech_dataset:", args.speech_dataset)
    print("visual_encoder:", args.visual_encoder)
    print("ckpt_path:", args.ckpt_path)
    print("test_condition:", args.test_condition)
    print("test_snr:", args.test_snr)
    print("embedding_size:", embedding_size)
    print("batch_size:", args.batch_size)
    print("num_workers:", args.num_workers)
    print("Input data:", os.path.join(speech_folder_path, f"dev/mixed_wav/{args.test_condition}/{args.test_snr}"))
    print("Output data:", os.path.join(speech_folder_path, f"dev/enhanced_wav/{args.visual_encoder}/{args.test_condition}/{args.test_snr}"))

    test(
        speech_dataset=args.speech_dataset,
        visual_encoder=args.visual_encoder,
        ckpt_path=args.ckpt_path,
        test_condition=args.test_condition,
        test_snr=args.test_snr,
        embedding_size=embedding_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
