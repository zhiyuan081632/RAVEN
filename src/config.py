import os

###############################################################
################# CHANGE YOUR PATH ############################
###############################################################

#TODO: replace with your own project root and data folder paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RAVEN/

# 支持多个语音数据集字典
# 格式: {数据集名: 数据路径}
SPEECH_DATASETS = {
    "VoxCeleb2": "/mnt/e/data/VoxCeleb2",
    "ChineseLips": "/mnt/e/data/ChineseLips",
    # 可在此添加更多数据集，如:
    # "LRS3": "/mnt/e/data/LRS3",
}

# 默认使用的数据集
DEFAULT_SPEECH_DATASET = "ChineseLips"
SPEECH_FOLDER_PATH = SPEECH_DATASETS[DEFAULT_SPEECH_DATASET]

MUSAN_FOLDER_PATH = "/mnt/e/data/MUSAN"

##############################################################
################## OVERALL CONFIGURATION #####################
##############################################################

embedding_size_dict = {
    "VSRiW": 512,
    "TalkNet": 512,
    "Loconet": 512,
    "AVHuBERT": 768,
    "VSRiW_TalkNet_concatenate": 1024,
    "VSRiW_TalkNet_addition": 512,
    "AVHuBERT_TalkNet_concatenate": 1280,
    "AVHuBERT_VSRiW_concatenate": 1280,
    "VSRiW_LRS3": 512,
    
}

VISUAL_ENCODER = "VSRiW_TalkNet_concatenate"
EMBEDDING_SIZE = embedding_size_dict[VISUAL_ENCODER]

AVHUBERT_PATH = os.path.join(PROJECT_ROOT, "av_hubert/avhubert")
VSRIW_PATH = os.path.join(PROJECT_ROOT, "Visual_Speech_Recognition_for_Multiple_Languages")
TALKNET_PATH = os.path.join(PROJECT_ROOT, "TalkNet_ASD")
LOCONET_PATH = os.path.join(PROJECT_ROOT, "LoCoNet_ASD")

SAMPLING_RATE = 16000
WINDOW_SIZE = 400
HOP_SIZE = 160
N_FFT = 512
P = 0.3

##############################################################
################# TRAINING CONFIGURATION #####################
##############################################################


DATE="20260311"
VERSION_NAME = f"{VISUAL_ENCODER}_5layer_{DATE}"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, f"src/checkpoints/{VERSION_NAME}")

# REPLACE WITH YOUR STORED MODEL CHECKPOINT
CKPT_PATH=os.path.join(PROJECT_ROOT, "src/checkpoints/path/to/checkpoint.ckpt")

TRAINING_LOWER_SNR = -10
TRAINING_UPPER_SNR = 10



##############################################################
################## TESTING CONFIGURATION #####################
##############################################################

# TEST_CONDITION available options:
# "one_interfering_speaker", "three_interfering_speakers", "noise_only"
TEST_CONDITION = "noise_only"
TEST_SNR = -10

TEST_VISUAL_ENCODERS = ["VSRiW", "TalkNet", "Loconet", "AVHuBERT", "AVHuBERT_TalkNet_concatenate", "AVHuBERT_VSRiW_concatenate", "VSRiW_TalkNet_concatenate"]
TEST_ALL_CONDITIONS = ["noise_only", "one_interfering_speaker", "three_interfering_speakers"]
TEST_ALL_SNRs = [-10, -5, 0, "mixed"]
