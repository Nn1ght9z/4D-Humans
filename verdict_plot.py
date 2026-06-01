import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
import os

def generate_baseline_aligned_verdict(npz_dir, output_path):
    print(f"[*] 启动终极物理判决：加载 6 组张量 (启用解剖学基线消除协议)...")
    file_paths = glob.glob(os.path.join(npz_dir, "dataset_set*.npz"))
    
    standard_tte_grid = np.linspace(0.0, 1.0, 1000)
    records = []
    
    for subject_idx, path in enumerate(file_paths):
        data = np.load(path)
        tte_raw = data['tte']
        her_raw = data['features'][:, 0]
        
        # 强制空间对齐
        interpolator = interp1d(tte_raw, her_raw, kind='linear', bounds_error=False, fill_value='extrapolate')
        her_aligned = interpolator(standard_tte_grid)
        
        # 🚨 修复 1: 解剖学基线对齐 (Anatomical Baseline Alignment)
        # 计算 TTE < 0.1 期间的均值作为个体物理基准
        baseline_mask = standard_tte_grid < 0.1
        her_baseline = np.mean(her_aligned[baseline_mask])
        
        # 计算相对于起跑状态的恶化比率 (Degradation Ratio)
        her_normalized = her_aligned / her_baseline
        
        for t, h_norm in zip(standard_tte_grid, her_normalized):
            records.append({
                'TTE': t,
                'HER_Ratio': h_norm,
                'Set_ID': subject_idx
            })
            
    df = pd.DataFrame(records)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制统计均值与 95% 置信区间
    sns.lineplot(
        data=df, x='TTE', y='HER_Ratio', 
        errorbar=('ci', 95), color='tab:red', linewidth=2.5, ax=ax
    )
    
    ax.set_xlim(0, 1.0)
    # 添加绝对基线 (y=1.0) 参考线
    ax.axhline(y=1.0, color='gray', linestyle='-.', alpha=0.5)
    ax.axvline(x=0.85, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.86, ax.get_ylim()[1]*0.9, 'Critical Phase\n(TTE > 0.85)', color='black', fontweight='bold')
    
    ax.set_title('The Ultimate Physical Verdict: Baseline-Aligned Kinematic Collapse (N=6)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Normalized Exhaustion Time (TTE)', fontweight='bold', fontsize=12)
    ax.set_ylabel('HER Degradation Ratio (Current / Baseline)', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[+] 基线消除完毕，统计学判决图谱固化至: {output_path}")

if __name__ == "__main__":
    generate_baseline_aligned_verdict("/root/autodl-tmp/", "/root/autodl-tmp/Ultimate_Physical_Verdict.png")