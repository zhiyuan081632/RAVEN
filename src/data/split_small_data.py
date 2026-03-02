import pandas as pd

full_data_path = "data/split_full.parquet"
small_data_path = "data/split_small.parquet"

# 读原始 split
df = pd.read_parquet(full_data_path)

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

# 覆盖写回 split.parquet
df_small.to_parquet(small_data_path)

# 同步生成 split.csv，供视频特征提取脚本使用
# 这里只保留 train/val/test 三个 split 的 audio_fp
split_for_csv = df_small[df_small["split"].isin(["train", "val", "test"])]["audio_fp"]
split_for_csv.to_csv("data/split_small.csv", index=False, header=False)

print("Extracting small data split from full data complete!")
print("Train rows:", len(df_small[df_small["split"] == "train"]))
print("Val rows:", len(df_small[df_small["split"] == "val"]))
print("Test rows:", len(df_small[df_small["split"] == "test"]))