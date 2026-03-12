import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def convert_single_mpg(args):
    """单个文件转换函数（必须是顶层函数才能被 pickle）"""
    mpg_file, dst_dir, src_path = args
    try:
        rel_path = mpg_file.relative_to(src_path)
    except ValueError:
        return False, f"⚠️ 跳过不在源目录下的文件: {mpg_file}"

    mp4_file = Path(dst_dir) / rel_path.with_suffix('.mp4')
    mp4_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-i', str(mpg_file),
        '-c:v', 'mpeg4',
        '-qscale:v', '5',
        '-c:a', 'aac',
        '-b:a', '128k',
        str(mp4_file)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, f"✅ {mpg_file} → {mp4_file}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else '未知错误'
        return False, f"❌ {mpg_file} - {error_msg[:300]}..."
    except Exception as ex:
        return False, f"⚠️ {mpg_file} - 异常: {ex}"

def convert_mpg_to_mp4_parallel(src_dir, dst_dir, max_workers=None):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 收集所有待处理文件
    tasks = []
    for mpg_file in src_path.rglob('*'):
        if mpg_file.is_file() and mpg_file.suffix.lower() in ('.mpg', '.mpeg'):
            tasks.append((mpg_file, dst_dir, src_path))

    if not tasks:
        print("未找到任何 .mpg 或 .mpeg 文件。")
        return

    print(f"共发现 {len(tasks)} 个 MPG 文件，开始并行转换（进程数: {max_workers or '自动'}）...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single_mpg, task) for task in tasks]
        for future in as_completed(futures):
            ok, msg = future.result()
            print(msg)
            if ok:
                success_count += 1

    print(f"\n🎉 转换完成！成功: {success_count}/{len(tasks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行转换 MPG 到 MP4（保持目录结构）")
    parser.add_argument("src_dir", help="源目录（包含 .mpg 文件）")
    parser.add_argument("dst_dir", help="目标目录（生成 .mp4 文件）")
    parser.add_argument("-j", "--jobs", type=int, default=None,
                        help="最大并发进程数（默认为 CPU 核数）")
    args = parser.parse_args()

    convert_mpg_to_mp4_parallel(args.src_dir, args.dst_dir, max_workers=args.jobs)