import os
import argparse
import glob
import pandas as pd
from tqdm import tqdm
try:
    import fitparse
except ImportError:
    raise ImportError("[-] 缺少核心依赖，请先执行: pip install fitparse pandas")

def parse_single_fit_to_csv(fit_file_path, output_csv_path):
    """
    底层解析算子：从 Garmin 原始二进制协议中剥离 1Hz 生物力学张量。
    时间复杂度: O(M)，M 为传感器数据报文长度。
    """
    fitfile = fitparse.FitFile(fit_file_path)
    records = []
    
    # 1. 遍历并解码连续的传感器数据帧
    for record in fitfile.get_messages('record'):
        data = {}
        for record_data in record:
            if record_data.name in ['timestamp', 'cadence', 'stance_time', 'vertical_oscillation']:
                data[record_data.name] = record_data.value
        
        # 仅保留包含时间戳的有效物理帧
        if 'timestamp' in data:
            records.append(data)
            
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    
    if df.empty:
        print(f"[-] 警告: {fit_file_path} 中未提取到有效的时间序列数据。")
        return False

    # 2. 构建连续的相对时间坐标系 (t=0 为记录启动瞬间)
    df['Time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    # 3. 维度映射与单位对齐 (严格契合 Step 2 接口标准)
    # Garmin 的 cadence 默认是单腿步频，需 x2 转化为全周期 SPM
    df['Cadence_spm'] = df.get('cadence', pd.Series([float('nan')]*len(df))) * 2
    df['GCT_ms'] = df.get('stance_time', float('nan'))
    df['VO_mm'] = df.get('vertical_oscillation', float('nan'))
    
    # 4. 剔除无效帧并固化张量
    # 仅保留完整包含三大核心生物力学标量的行，防止下游三次样条插值遭遇 NaN 奇点
    final_df = df[['Time_s', 'Cadence_spm', 'GCT_ms', 'VO_mm']].dropna()
    
    if final_df.empty:
        print(f"[-] 警告: {fit_file_path} 缺失 GCT 或 VO 核心动态数据 (请确认是否佩戴了 HRM-Pro 或 RD Pod)。")
        return False
        
    final_df.to_csv(output_csv_path, index=False)
    return len(final_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency-Mamba 物理 GT 批量提取器")
    parser.add_argument("--input_dir", type=str, required=True, help="存放 6 个原始 .fit 文件的目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="提取后的 .csv 矩阵输出目录")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fit_files = glob.glob(os.path.join(input_dir, "*.fit"))
    if not fit_files:
        raise FileNotFoundError(f"[-] 致命错误：在 {input_dir} 下未探测到任何 .fit 文件。")

    print(f"[*] 探测到 {len(fit_files)} 组物理原始文件，启动批量解析高炉...")
    
    success_count = 0
    for fit_path in tqdm(fit_files, desc="Parsing FIT -> CSV"):
        file_name = os.path.basename(fit_path)
        csv_name = file_name.replace('.fit', '.csv')
        csv_path = os.path.join(output_dir, csv_name)
        
        valid_frames = parse_single_fit_to_csv(fit_path, csv_path)
        if valid_frames:
            success_count += 1
            
    print(f"[+] 物理数据提纯完毕。成功固化 {success_count}/{len(fit_files)} 组 1Hz CSV 张量，存入: {output_dir}")