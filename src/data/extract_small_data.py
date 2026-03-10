'''
从data/split.csv中抽取前1000条训练数据
从data/VoxCeleb2_val_1000_fps.txt中抽取1000条验证数据
从data/VoxCeleb2_test_1000_fps.txt中抽取1000条测试数据
'''

import pandas as pd

# 读原始 split（三列：audio_fp, video_fp, split）
df = pd.read_csv("data/split_full.csv") # 读取CSV文件

# 1) 只保留前 1000 条 train 数据
train_mask = df["split"] == "train"
df_train = df[train_mask].head(1000)

# 2) 用 TXT 中的文件名精确过滤 val（一般是 1000 条）
val_list = pd.read_csv("data/VoxCeleb2_val_1000_fps.txt", header=None)[0]
val_mask = df["split"] == "val"
df_val = df[val_mask & df["audio_fp"].isin(val_list)]

# 3) 用 TXT 中的文件名精确过滤 test（一般是 1000 条）
test_list = pd.read_csv("data/VoxCeleb2_test_1000_fps.txt", header=None)[0]
test_mask = df["split"] == "test"
df_test = df[test_mask & df["audio_fp"].isin(test_list)]

# 4) 其它 split（如果还有的话）保持不变
df_others = df[~(train_mask | val_mask | test_mask)]

# 合并回去，生成缩减版 split
df_small = pd.concat([df_train, df_val, df_test, df_others], ignore_index=True)

# 生成缩减版 split_small.csv
df_small.to_csv("data/split_small.csv", index=False, header=True)

print("Extracting small data split from full data complete!")
print("Train rows:", len(df_small[df_small["split"] == "train"]))
print("Val rows:", len(df_small[df_small["split"] == "val"]))
print("Test rows:", len(df_small[df_small["split"] == "test"]))