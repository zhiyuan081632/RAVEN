import os

###############################################################
################# PROJECT PATH ############################
###############################################################

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # RAVEN/


##############################################################
################ DATA LIST CONFIGURATION #####################
##############################################################
# list 文件路径（data/ 目录下），每个文件每行一个绝对路径
# speech: mp4 绝对路径，音频路径自动由 /mp4/ -> /wav/, .mp4 -> .wav 推导

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 训练集 speech list（可配置多个）
SPEECH_TRAIN_LISTS = [
    os.path.join(_DATA_DIR, "VoxCeleb2_train_1000.txt"),
    os.path.join(_DATA_DIR, "GRID_train_1000.txt"),
    # "/mnt/e/data/VoxCeleb2/dev/VoxCeleb2_train_list.txt",
]
# 验证集 speech list
SPEECH_VAL_LISTS = [
    os.path.join(_DATA_DIR, "VoxCeleb2_val_1000.txt"),
]
# 测试集 speech list
SPEECH_TEST_LISTS = [
    os.path.join(_DATA_DIR, "VoxCeleb2_test_1000.txt"),
]

# 噪声 list（按 split 分组）
import glob as _glob
NOISE_LISTS = {
    "train": sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_noise_train*.txt"))),
    "val":   sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_noise_va*.txt"))),
    "test":  sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_noise_tes*.txt"))),
}
# 音乐 list（按 split 分组）
MUSIC_LISTS = {
    "train": sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_music_trai*.txt"))),
    "val":   sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_music_va*.txt"))),
    "test":  sorted(_glob.glob(os.path.join(_DATA_DIR, "musan_music_tes*.txt"))),
}

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
