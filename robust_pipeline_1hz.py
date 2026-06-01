import os
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import scipy.signal as signal
from scipy.interpolate import CubicSpline
import scipy.ndimage as ndimage

# 强制切断 CPU 底层多线程争用
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- HMR2 依赖 ---
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

class FrequencyMambaPipeline:
    """
    Frequency-Mamba 连续生物力学评估网络的物理特征提取器
    """
    def __init__(self, batch_size=256, device_str='cuda'):
        self.device = torch.device(device_str) if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        print("[*] 正在加载 HMR2 权重与检测器拓扑...")
        self.model, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device).eval()

    def _calc_sagittal_batch(self, joints_3d_batch):
        l_hip, r_hip = joints_3d_batch[:, 12, :], joints_3d_batch[:, 9, :]
        r_knee, r_ankle = joints_3d_batch[:, 10, :], joints_3d_batch[:, 11, :]
        
        normal = l_hip - r_hip
        normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-8)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3).expand(joints_3d_batch.shape[0], -1)
        y_proj = y_axis - torch.sum(y_axis * normal, dim=1, keepdim=True) * normal

        def get_angle(top, bottom):
            vec = bottom - top
            proj = vec - torch.sum(vec * normal, dim=1, keepdim=True) * normal
            cos_val = torch.sum(proj * y_proj, dim=1) / (torch.norm(proj, dim=1) * torch.norm(y_proj, dim=1) + 1e-8)
            return 90.0 - torch.rad2deg(torch.acos(torch.clamp(cos_val, -1.0, 1.0)))

        return get_angle(r_hip, r_knee).cpu().numpy(), get_angle(r_knee, r_ankle).cpu().numpy()

    def _extract_spatial_tensors(self, video_path):
        """Pass A: O(1) 静态锚定与批量 3D 骨架抽取"""
        print(f"[*] Pass A: 抽取空间张量 -> {os.path.basename(video_path)}")
        cfg_path = Path(__import__('hmr2').__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        det_cfg = LazyConfig.load(str(cfg_path))
        det_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        detector = DefaultPredictor_Lazy(det_cfg)

        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret: raise ValueError("[-] 无法读取视频流。")
        
        det_out = detector(first_frame)
        valid = (det_out['instances'].pred_classes == 0) & (det_out['instances'].scores > 0.5)
        boxes = det_out['instances'].pred_boxes.tensor[valid].cpu().numpy()
        target_box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
        del detector 

        center = np.array([(target_box[0] + target_box[2]) / 2.0, (target_box[1] + target_box[3]) / 2.0])
        scale = max(target_box[2] - target_box[0], target_box[3] - target_box[1]) / 200.0 * 1.2
        src_w = scale * 200.0
        src = np.array([center + scale * np.array([0,0]), center + np.array([0, src_w*-0.5]), center + np.array([-src_w*-0.5, 0])], dtype=np.float32)
        dst = np.array([[128,128], [128, 0], [0, 128]], dtype=np.float32)
        trans = cv2.getAffineTransform(src, dst)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        t_all, s_all, batch_img, batch_c, batch_s = [], [], [], [],[]

        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc="Affine Extractor"):
                ret, frame = cap.read()
                if not ret: break
                
                img = cv2.warpAffine(frame, trans, (256, 256), flags=cv2.INTER_LINEAR)
                img = (img[:, :, ::-1].astype(np.float32) / 255.0 - mean) / std
                batch_img.append(img.transpose(2, 0, 1))
                batch_c.append(center)
                batch_s.append([scale])
                
                if len(batch_img) == self.batch_size:
                    out = self.model({'img': torch.tensor(np.array(batch_img), device=self.device),
                                      'box_center': torch.tensor(np.array(batch_c), dtype=torch.float32, device=self.device),
                                      'box_size': torch.tensor(np.array(batch_s), dtype=torch.float32, device=self.device)})
                    t_ang, s_ang = self._calc_sagittal_batch(out['pred_keypoints_3d'])
                    t_all.extend(t_ang)
                    s_all.extend(s_ang)
                    batch_img, batch_c, batch_s = [], [],[]
                    
            if len(batch_img) > 0:
                out = self.model({'img': torch.tensor(np.array(batch_img), device=self.device),
                                  'box_center': torch.tensor(np.array(batch_c), dtype=torch.float32, device=self.device),
                                  'box_size': torch.tensor(np.array(batch_s), dtype=torch.float32, device=self.device)})
                t_ang, s_ang = self._calc_sagittal_batch(out['pred_keypoints_3d'])
                t_all.extend(t_ang)
                s_all.extend(s_ang)

        cap.release()
        return np.array(t_all), np.array(s_all), fps

    def _extract_spectral_stft(self, thigh_seq, shank_seq, fps, window_sec=10.0, step_sec=1.0):
        """Pass B: 基于严谨 STFT 的频域特征提取 (完全阻断 Zero-Padding)"""
        print(f"[*] Pass B: 提取频域流形 (STFT 边界约束生效)...")
        win_f = int(window_sec * fps)
        step_f = int(step_sec * fps)
        noverlap = win_f - step_f
        
        # 🚨 指令 1 & 2: 强制 boundary=None 且 padded=False，确保只有满窗数据才会输出计算
        # nfft 显式绑定至 nperseg，彻底阻断 FFT 内部零填充引入的频谱截断伪影
        f, t_sec, Zxx_t = signal.stft(thigh_seq, fs=fps, window='hann', nperseg=win_f,
                                      noverlap=noverlap, nfft=win_f, detrend='linear',
                                      boundary=None, padded=False)
        _, _, Zxx_s = signal.stft(shank_seq, fs=fps, window='hann', nperseg=win_f,
                                  noverlap=noverlap, nfft=win_f, detrend='linear',
                                  boundary=None, padded=False)

        Pxx_t = np.abs(Zxx_t)**2
        Pxy = Zxx_t * np.conj(Zxx_s)

        vis_t = t_sec
        feats = []
        # 记录每窗口是否通过能量门控 (插值前)，供下游识别物理死区
        valid_flags = []
        valid_idx = np.where((f >= 1.2) & (f <= 3.5))[0]

        # 自适应能量阈值：基于全信号 PSD 统计而非硬编码 1e-4
        # 取所有窗口总能量的 0.1% 分位作为最低物理显著性门槛
        window_energies = np.sum(Pxx_t, axis=0)
        nonzero_energies = window_energies[window_energies > 0]
        if len(nonzero_energies) > 0:
            energy_threshold = max(np.percentile(nonzero_energies, 1) * 0.1, 1e-8)
        else:
            energy_threshold = 1e-8

        def get_E(target_f, Pxx_col, freq_axis):
            idx = np.where((freq_axis >= target_f - 0.15) & (freq_axis <= target_f + 0.15))[0]
            return np.sum(Pxx_col[idx]) if len(idx) > 0 else 0.0

        for i in range(len(vis_t)):
            window_energy = np.sum(Pxx_t[:, i])
            if window_energy < energy_threshold or len(valid_idx) == 0:
                feats.append([np.nan, np.nan])
                valid_flags.append(False)
                continue

            f0_idx = valid_idx[np.argmax(Pxx_t[valid_idx, i])]
            f0 = f[f0_idx]

            E_fund = get_E(f0, Pxx_t[:, i], f)
            E_harm = get_E(f0 * 2, Pxx_t[:, i], f) + get_E(f0 * 3, Pxx_t[:, i], f)

            # 绝对除零保护 + 生物力学边界约束: 追踪失败造成的高频乱码不得进入归一化
            her = np.clip(E_harm / (E_fund + 1e-8), 0.0, 5.0)
            phase_deg = np.degrees(np.angle(Pxy[np.argmin(np.abs(f - f0)), i]))
            feats.append([her, phase_deg])
            valid_flags.append(True)

        feats_arr = np.array(feats)
        valid_flags = np.array(valid_flags)

        if np.any(np.isfinite(feats_arr[:, 1])):
            valid_phase = np.isfinite(feats_arr[:, 1])
            feats_arr[valid_phase, 1] = np.rad2deg(np.unwrap(np.deg2rad(feats_arr[valid_phase, 1])))

        # 仅在有效窗口上执行插值与滤波，避免 NaN 向物理死区反向传播
        if np.sum(valid_flags) >= 2:
            df_feats = pd.DataFrame(feats_arr)
            feats_arr = df_feats.interpolate(method='linear', limit_direction='both').values
        feats_arr[:, 0] = ndimage.median_filter(feats_arr[:, 0], size=3, mode='nearest')
        feats_arr[:, 1] = ndimage.median_filter(feats_arr[:, 1], size=3, mode='nearest')

        return vis_t, feats_arr, valid_flags

    def process_and_save(self, video_path, imu_csv_path, output_npz_path):
        """执行端到端对齐，三相基线归一化 (HER + 相位 + 首窗口鲁棒锚定)，输出 Mamba 就绪张量"""
        t_seq, s_seq, fps = self._extract_spatial_tensors(video_path)
        vis_t, vis_f, vis_valid = self._extract_spectral_stft(t_seq, s_seq, fps)

        print(f"[*] Step 2: 时序对齐与三相基线归一化...")
        df = pd.read_csv(imu_csv_path)
        imu_t = df['Time_s'].values
        imu_t_max = imu_t.max()

        valid_mask = (vis_t >= imu_t.min()) & (vis_t <= imu_t_max)
        aligned_t = vis_t[valid_mask]
        aligned_f = vis_f[valid_mask]
        aligned_valid = vis_valid[valid_mask]

        if len(aligned_t) == 0:
            raise ValueError(f"[-] 严重错误：视觉特征与 IMU 数据在时间轴上没有交集。")

        # ═══════════════════════════════════════════════
        # 三相基线归一化 (Phase 0: 首窗口物理有效性防御)
        # ═══════════════════════════════════════════════

        # 前向扫描：跳过被能量门控标记为无效的窗口，找到第一个物理上真正的有效帧
        first_valid_idx = 0
        for idx in range(min(len(aligned_f), max(1, len(aligned_f) // 10))):
            if aligned_valid[idx] and not np.isnan(aligned_f[idx, 0]) and aligned_f[idx, 0] > 1e-6:
                first_valid_idx = idx
                break

        # 取前 5 个有效窗口中值作为基线 (单帧在噪声下不可靠)
        valid_indices = [
            i for i in range(first_valid_idx, min(first_valid_idx + 10, len(aligned_f)))
            if aligned_valid[i] and not np.isnan(aligned_f[i, 0]) and aligned_f[i, 0] > 1e-6
        ]
        baseline_indices = valid_indices[:5] if len(valid_indices) >= 5 else valid_indices
        if len(baseline_indices) < 1:
            raise ValueError("[-] 前 10 个窗口中无任何物理有效帧，视频数据已彻底毁损。")

        baseline_her = np.median(aligned_f[baseline_indices, 0])
        baseline_phase = np.median(aligned_f[baseline_indices, 1])

        print(f"    → 基线锚定于窗口 #{baseline_indices[0]}..{baseline_indices[-1]}"
              f" (共 {len(baseline_indices)} 帧), HER_baseline={baseline_her:.4f}, Phase_baseline={baseline_phase:.1f}°")

        # Phase 1: TTE 平移 — 强制第一个有效特征点锚定为 TTE=0
        t0 = aligned_t[first_valid_idx]
        aligned_tte = (aligned_t - t0) / (imu_t_max - t0 + 1e-8)

        # Phase 2: HER 基线归一化 — 使 TTE=0 时 HER ≡ 1.0
        aligned_f[:, 0] = np.clip(aligned_f[:, 0] / (baseline_her + 1e-8), 0.0, 5.0)

        # Phase 3: 相位角基线消除 — 测量相对于初始状态的协调漂移量 Δθ
        aligned_f[:, 1] = aligned_f[:, 1] - baseline_phase

        # 强制锚定：消除浮点运算残差
        aligned_f[first_valid_idx, 0] = 1.0
        aligned_f[first_valid_idx, 1] = 0.0
        aligned_tte[first_valid_idx] = 0.0

        gct_spline = CubicSpline(imu_t, df['GCT_ms'].values)(aligned_t)
        vo_spline = CubicSpline(imu_t, df['VO_mm'].values)(aligned_t)
        spm_spline = CubicSpline(imu_t, df['Cadence_spm'].values)(aligned_t)

        np.savez(output_npz_path,
                 features=aligned_f,
                 tte=aligned_tte,
                 gct=gct_spline,
                 vo=vo_spline,
                 spm=spm_spline)
        print(f"[+] 三相归一化完毕。张量包已封印至: {output_npz_path}")

# ================= 批处理执行入口 =================
if __name__ == "__main__":
    # pipeline = FrequencyMambaPipeline(batch_size=256)
    # pipeline.process_and_save("video.mp4", "imu.csv", "out.npz")
    pass