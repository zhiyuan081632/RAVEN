import os
import requests
from tqdm import tqdm
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
 
# 下载GRID_Dataset数据集的URL
BASE_URL = "https://spandh.dcs.shef.ac.uk/gridcorpus/"
 
SAVE_DIR = "/mnt/d/data/GRID/zip/" # 下载保存目录
UNZIP_DIR = "/mnt/d/data/GRID/unzip/" # 解压目录

# 检查目录是否存在，如果不存在，则创建该目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(UNZIP_DIR):
    os.makedirs(UNZIP_DIR)


def extract_archive(archive_path, extract_to):
    """解压tar或zip文件到指定目录，保持原目录结构"""
    if not os.path.exists(archive_path):
        print(f"归档文件 {archive_path} 不存在，跳过解压。")
        return
    
    # 获取相对于SAVE_DIR的路径，用于在UNZIP_DIR中保持相同结构
    rel_path = os.path.relpath(os.path.dirname(archive_path), SAVE_DIR)
    # final_extract_path = os.path.join(extract_to, rel_path)
    final_extract_path = extract_to
    
    # 创建解压目标目录
    os.makedirs(final_extract_path, exist_ok=True)
    
    try:
        # 判断文件类型并解压
        if archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            print(f"正在解压 {os.path.basename(archive_path)} 到 {final_extract_path}...")
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(path=final_extract_path)
            print(f"✓ {os.path.basename(archive_path)} 解压完成")
            
        elif archive_path.endswith('.zip'):
            print(f"正在解压 {os.path.basename(archive_path)} 到 {final_extract_path}...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(final_extract_path)
            print(f"✓ {os.path.basename(archive_path)} 解压完成")
        else:
            print(f"不支持的归档格式: {archive_path}")
            
    except Exception as e:
        print(f"解压 {archive_path} 时出错: {e}")


def download_file(url, save_path):
    """下载文件并显示进度条"""
    # 检查文件是否已经存在
    if os.path.exists(save_path):
        print(f"{os.path.basename(save_path)} 已存在，跳过下载。")
        return
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        total_size = int(response.headers.get('content-length', 0))

        with open(save_path, 'wb') as file, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(len(data))
                file.write(data)

    except requests.exceptions.RequestException as e:
        print(f"下载 {url} 时出错: {e}")


# 线程安全的进度条锁
progress_lock = threading.Lock()

def extract_archive_safe(archive_path, extract_to):
    """线程安全的解压函数"""
    extract_archive(archive_path, extract_to)
    with progress_lock:
        print(f"[{threading.current_thread().name}] 完成解压: {os.path.basename(archive_path)}")

# 主循环 - 并发解压
for i in range(1, 35):
    if i == 21:
        continue  # 跳过第21个说话者

    video1 = f"s{i}/video/s{i}.mpg_6000.part1.tar"
    video2 = f"s{i}/video/s{i}.mpg_6000.part2.tar"
    txt_label = f"s{i}/align/s{i}.tar"

    # 下载的文件路径
    file_paths = [BASE_URL + video1, BASE_URL + video2, BASE_URL + txt_label]
    # 保存文件的路径
    save_paths = [SAVE_DIR + video1, SAVE_DIR + video2, SAVE_DIR + txt_label]

    # # 开始批量下载（保持串行）
    # for file_path, save_path in zip(file_paths, save_paths):
    #     # 创建保存文件的目录（如果不存在）
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     # 下载文件
    #     download_file(file_path, save_path)

    # print(f"{i}号文件下载完成！")
    
    # 并发解压所有文件
    print(f"\n开始并发解压 s{i} 的文件...")
    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="Extract") as executor:
        futures = [
            executor.submit(extract_archive_safe, save_path, UNZIP_DIR) 
            for save_path in save_paths
        ]
        # 等待所有解压任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取异常
            except Exception as e:
                print(f"解压出错: {e}")
    
    print(f"s{i} 所有文件解压完成！\n")

print("\n" + "="*60)
print("所有下载和解压任务完成！")
print(f"下载文件位置: {SAVE_DIR}")
print(f"解压文件位置: {UNZIP_DIR}")
print("="*60)
