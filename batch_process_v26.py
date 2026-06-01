"""
v26 批量处理脚本：自动发现 video+CSV 配对，执行三相基线归一化，输出 Mamba 训练就绪张量。
必须在 hmr2 conda 环境中运行，且需要 CUDA GPU。
用法: source activate hmr2 && python3 batch_process_v26.py
"""
import os, re, time
import numpy as np

# 强制清空可能残留的线程限制
for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.pop(var, None)

# ═══ 配置 ═══
DATA_ROOT = '/root/autodl-tmp'
VIDEO_DIR = os.path.join(DATA_ROOT, 'video_raw')
CSV_DIR = os.path.join(DATA_ROOT, 'imu_csv')
OUT_DIR = '/root/4D-Humans/aligned_datasets_v26'
BATCH_SIZE = 256  # RTX 3090 24GB 可开 256

os.makedirs(OUT_DIR, exist_ok=True)


def _natural_key(name):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', name)]


def discover_pairs():
    videos = {os.path.splitext(f)[0].replace('video_', ''): f
              for f in os.listdir(VIDEO_DIR)
              if f.startswith('video_') and f.lower().endswith('.mp4')}
    csvs = {os.path.splitext(f)[0].replace('imu_', ''): f
            for f in os.listdir(CSV_DIR)
            if f.startswith('imu_') and f.lower().endswith('.csv')}

    pairs = []
    for key in sorted(set(videos) & set(csvs), key=_natural_key):
        pairs.append((key, videos[key], csvs[key]))

    missing_csv = sorted(set(videos) - set(csvs), key=_natural_key)
    missing_video = sorted(set(csvs) - set(videos), key=_natural_key)
    for key in missing_csv:
        print(f"[-] 缺失 CSV: imu_{key}.csv, 跳过 video_{key}.mp4")
    for key in missing_video:
        print(f"[-] 缺失视频: video_{key}.mp4, 跳过 imu_{key}.csv")
    return pairs


def main():
    print("=" * 64)
    print("  v26 批量特征提取管线")
    print("  三相基线归一化: TTE=0, HER≡1.0, Δθ≡0°")
    print("=" * 64)

    from robust_pipeline_1hz import FrequencyMambaPipeline
    pipeline = FrequencyMambaPipeline(batch_size=BATCH_SIZE)

    pairs = discover_pairs()
    if not pairs:
        raise FileNotFoundError(f"未在 {VIDEO_DIR} 和 {CSV_DIR} 找到匹配的 video_*.mp4 / imu_*.csv")

    print(f"  发现 {len(pairs)} 组 video+CSV 配对")

    for idx, (session_id, video_name, csv_name) in enumerate(pairs, 1):
        video_path = os.path.join(VIDEO_DIR, video_name)
        csv_path = os.path.join(CSV_DIR, csv_name)
        output_path = os.path.join(OUT_DIR, f'aligned_set{session_id}.npz')

        print(f"\n{'='*48}")
        print(f"  [{idx}/{len(pairs)}] 处理: {video_name} + {csv_name}")
        print(f"{'='*48}")

        t0 = time.time()
        try:
            pipeline.process_and_save(video_path, csv_path, output_path)
            elapsed = time.time() - t0
            print(f"  ✓ 完成 ({elapsed:.0f}s) → {output_path}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

        # 审计输出质量
        data = np.load(output_path)
        her = data['features'][:, 0]
        phase = data['features'][:, 1]
        tte = data['tte']
        gct = data['gct']

        print(f"  审计: session={session_id} | video={video_name} | csv={csv_name} | "
              f"TTE=[{tte[0]:.4f}, {tte[-1]:.4f}] | "
              f"HER@TTE=0={her[0]:.4f} | HER=[{np.nanmin(her):.2f}, {np.nanmax(her):.2f}] | "
              f"Phase@TTE=0={phase[0]:.1f}° | GCT=[{gct.min():.0f}, {gct.max():.0f}]ms | "
              f"frames={len(tte)}")

    print(f"\n{'='*64}")
    print(f"  全部处理完毕。输出目录: {OUT_DIR}/")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
