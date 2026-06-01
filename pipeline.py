"""
Frequency-Mamba: 基于频域流形与状态空间模型的连续疲劳评估网络
=====================================================================
架构: STFT频域特征 X_t(HER, Δθ) → Conv1d → Mamba SSM → ŷ(TTE)
核心卖点: O(N) 复杂度的选择性状态空间模型作为非线性卡尔曼滤波器，
         从充满异方差观测噪声的频域特征中回归平滑疲劳曲线。

数学注释:
  Mamba SSM 的核心递推:
    h_t = Â_t ⊙ h_{t-1} + B̂_t ⊙ x_t    (状态更新)
    y_t = C_t @ h_t                      (观测投影)
  其中 Â_t = exp(Δ_t ⊙ A), B̂_t = Δ_t ⊙ B_t
  A 为可学习对角状态矩阵 (对数空间), Δ_t 为输入依赖的步长。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os, time, warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# 0. 全局配置
# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'pipeline_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. 合成数据生成器 (生理先验注入)
# ═══════════════════════════════════════════════════════════════
def generate_synthetic_cohort(n_subjects=6, T=500, fs_vis=1.0,
                               noise_config='heteroscedastic'):
    """
    生成符合跑步生理学的合成频域特征时序数据。

    物理先验 (来自文献与真实数据观察):
      - HER 从 1.0 出发, 随疲劳呈指数/超线性上升趋势
      - 观测噪声随 HER 幅值放大 (异方差性, 视觉 Jitter 的内在属性)
      - Δθ 呈缓慢线性漂移, 反映关节间协调相位的渐进失配
      - TTE 是绝对平滑的单调递增归一化力竭时间

    Args:
        n_subjects: 模拟受试者数量
        T:         时间步数 (对应 1Hz 采样下的秒数)
        fs_vis:    视觉特征采样率 (Hz)
        noise_config: 'heteroscedastic' | 'homoscedastic' | 'clean'

    Returns:
        X: [n_subjects, T, 2] 频域特征 (channel 0=HER, channel 1=Δθ)
        Y: [n_subjects, T, 1] 归一化 TTE 标签
    """
    t = torch.linspace(0, 1, T).unsqueeze(0).repeat(n_subjects, 1)  # [S, T]

    # 受试者间生理变异参数 (模拟个体差异)
    fatigue_rate = 2.5 + 0.8 * torch.randn(n_subjects, 1)       # 疲劳速率 ∈ [1.7, 3.3]
    fatigue_rate = torch.clamp(fatigue_rate, 1.5, 3.5)
    her_onset = 1.0 + 0.05 * torch.randn(n_subjects, 1)          # 初始 HER ∈ [0.9, 1.1]
    her_onset = torch.clamp(her_onset, 0.8, 1.2)
    phase_drift_rate = 15.0 + 10.0 * torch.randn(n_subjects, 1)  # 相位漂移率 (°/归一化时间)
    phase_drift_rate = torch.clamp(phase_drift_rate, 2.0, 35.0)

    # --- HER 生成: 指数上升 + 异方差高斯噪声 ---
    # 底层物理: HER(t) = baseline + (exp(α·t) - 1) + ε(t), ε ~ N(0, σ²(t))
    # 异方差: σ(t) = σ_base * HER_clean(t), 模拟视觉追踪 Jitter 随运动幅度放大
    her_clean = her_onset + (torch.exp(fatigue_rate * t) - 1.0)  # [S, T]

    if noise_config == 'heteroscedastic':
        sigma_her = 0.3 * her_clean                                # 噪声标准差与信号幅值成正比
        her_noise = sigma_her * torch.randn(n_subjects, T)
    elif noise_config == 'homoscedastic':
        her_noise = 0.5 * torch.randn(n_subjects, T)
    else:  # clean
        her_noise = 0.02 * torch.randn(n_subjects, T)

    her = her_clean + her_noise
    her = torch.clamp(her, min=0.05)                               # 物理约束: HER > 0

    # --- Δθ 生成: 线性漂移 + 恒定方差噪声 ---
    phase_clean = phase_drift_rate * t                              # [S, T], 度
    phase_noise = 3.0 * torch.randn(n_subjects, T)                 # σ=3° 恒定噪声
    theta = phase_clean + phase_noise

    # --- 组合输入 ---
    X = torch.stack([her, theta], dim=-1)                           # [S, T, F=2]

    # --- TTE 标签: 平滑单调递增, 注入轻微受试者间变异 ---
    # Y(t) ∈ [0, 1], 严格单调递增的平滑 S 形曲线
    tau = 0.45 + 0.1 * torch.randn(n_subjects, 1)                  # 拐点位置, 受试者间变异
    tau = torch.clamp(tau, 0.3, 0.55)
    steepness = 4.0 + 1.5 * torch.randn(n_subjects, 1)
    steepness = torch.clamp(steepness, 2.5, 6.0)

    logistic = 1.0 / (1.0 + torch.exp(-steepness * (t - tau)))     # S 形基函数
    linear = t                                                      # 线性基函数
    # 混合: 70% logistic + 30% linear → 保持整体单调且两端平滑
    Y = (0.7 * logistic + 0.3 * linear).unsqueeze(-1)              # [S, T, 1]

    print(f"[数据] 合成 {n_subjects} 组受试者, T={T}, 噪声模式={noise_config}")
    print(f"       HER_clean 范围: [{her_clean.min():.2f}, {her_clean.max():.2f}]")
    print(f"       HER_noisy 范围: [{her.min():.2f}, {her.max():.2f}]")
    return X, Y


# ═══════════════════════════════════════════════════════════════
# 2. Mamba SSM 核心算子 (纯 PyTorch 实现)
# ═══════════════════════════════════════════════════════════════
class MambaSSM(nn.Module):
    """
    选择性状态空间模型 (Selective SSM) 的数学实现。

    离散化 (Zero-Order Hold):
        A_d = exp(Δ ⊙ A)                 # A ∈ R^{D, N}: 对角状态矩阵
        B_d = (Δ ⊙ B)                    # B ∈ R^{B, T, N}: 输入投影
        C   = Linear(x)                  # C ∈ R^{B, T, N}: 输出投影
        Δ   = softplus(Linear(x) + bias) # ∈ R^{B, T, D}: 输入依赖步长

    状态递推 (对角 SSM 的逐元素运算):
        h_t = A_d ⊙ h_{t-1} + B_d ⊙ x_t   (⊙ = element-wise)
        y_t = C @ h_t + D * x_t

    复杂度: O(T·D·N), 对于对角 SSM 等价于 O(T·D).

    NOTE: 此实现使用顺序扫描 (sequential scan)。生产部署时可替换为
          Mamba 官方的 parallel associative scan 以利用 GPU 并行性。
    """

    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, T_seq=500):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(d_model * expand)
        self.d_inner = d_inner
        self.T_seq = T_seq

        # 输入投影: x → expanded hidden
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D 深度可分离卷积 (局部时序平滑，Mamba 标配)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=False
        )
        self.act = nn.SiLU()

        # ═══ SSM 参数初始化 ═══
        # A_log ∈ [d_inner, d_state]
        # HiPPO 理论: 特征值幅度应在 [0.5/T, 0.5] 区间几何分布，
        # 确保 d_state 个通道覆盖从 ~1 步到 ~T 步的全频谱记忆范围。
        # 反例: 原 init log(1:16)=[0, 2.77] 全部为短记忆 (<2步)，导致梯度崩溃
        A_init = torch.tensor(
            np.log(np.geomspace(0.5 / max(T_seq, 1), 0.5, d_state)),
            dtype=torch.float32
        ).unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(A_init)

        self.D = nn.Parameter(torch.ones(d_inner))

        # Δ, B, C 投影
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        # dt bias: 使初始 Δ ≈ 1/T, 模型以"拷贝模式"起步 (长记忆优先)
        with torch.no_grad():
            self.dt_proj.bias.fill_(np.log(np.exp(1.0 / max(T_seq, 1)) - 1))
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        对角 SSM 的顺序扫描 (选择性扫描核心)。

        Args:
            u:     [B, L, D] 输入序列
            delta: [B, L, D] 步长
            A:     [D, N]   对数状态矩阵
            B:     [B, L, N] 输入转移向量
            C:     [B, L, N] 输出映射向量
            D:     [D]       直连残差

        Returns:
            y: [B, L, D] 输出序列
        """
        B, L, D = u.shape
        N = A.shape[1]

        # 离散化: A_d = -exp(A_log * Δ), 确保稳定 (负特征值 → 衰减记忆)
        # Mamba 使用 A_d = exp(Δ * A) 其中 A 为负 → exp(-|A|Δ) ∈ (0,1]
        A_d = -torch.exp(A) * delta.unsqueeze(-1)            # [B, L, D, N]
        A_d = torch.exp(A_d)                                   # [B, L, D, N] ∈ (0, 1]

        # B_d = Δ * B
        B_d = delta.unsqueeze(-1) * B                          # [B, L, N]

        # 顺序扫描 (可替换为并行关联扫描)
        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)  # [B, D, N]
        outputs = []

        for t in range(L):
            # h_t = A_d[:, t] ⊙ h_{t-1} + B_d[:, t] ⊙ u[:, t]  (广播: [B,D,N] * [B,D,1] → [B,D,N])
            h = A_d[:, t] * h + B_d[:, t] * u[:, t, :, None]
            # y_t = Σ_n C[:, t, n] * h[:, :, n]
            y_t = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1)    # [B, D]
            y_t = y_t + D * u[:, t]                               # 直连残差
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                           # [B, L, D]
        return y

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model] 输入特征
        Returns:
            out: [B, L, d_model]
        """
        B, L, _ = x.shape

        # 1. 输入投影 + 分支
        xz = self.in_proj(x)                                      # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)                           # 各 [B, L, d_inner]

        # 2. 1D 卷积 (局部时序平滑)
        x_conv = self.conv1d(x_proj.transpose(1, 2))              # [B, d_inner, L + d_conv-1]
        x_conv = x_conv[..., :L]                                   # 截断至原始长度
        x_conv = self.act(x_conv.transpose(1, 2))                 # [B, L, d_inner]

        # 3. SSM 参数生成
        delta = F.softplus(self.dt_proj(x_conv))                  # [B, L, d_inner]
        BC = self.x_proj(x_conv)                                   # [B, L, 2*d_state]
        B_ssm, C_ssm = BC.chunk(2, dim=-1)                        # 各 [B, L, d_state]

        # 4. 选择性扫描
        y_ssm = self._selective_scan(x_conv, delta, self.A_log,
                                      B_ssm, C_ssm, self.D)       # [B, L, d_inner]

        # 5. 门控 + 输出投影
        y = y_ssm * F.silu(z)                                      # [B, L, d_inner]
        out = self.out_proj(y)                                     # [B, L, d_model]
        return out


class MambaBlock(nn.Module):
    """Mamba 基础模块: LayerNorm → SSM → 残差连接"""
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, T_seq=500):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = MambaSSM(d_model, d_state, d_conv, expand, T_seq)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


# ═══════════════════════════════════════════════════════════════
# 3. Frequency-Mamba 完整网络
# ═══════════════════════════════════════════════════════════════
class FrequencyMamba(nn.Module):
    """
    频域流形 → 状态空间 → 疲劳回归

    架构:
      Input [B, T, F=2]
        ↓
      Conv1d Stem (局部频域纹理提取, kernel=5, padding=2)
        ↓
      LayerNorm + GELU
        ↓
      Mamba Blocks × n_layers (选择性状态空间时序建模)
        ↓
      LayerNorm → Linear Head → ŷ ∈ [0, 1]
    """

    def __init__(self, input_dim=2, d_model=64, d_state=16,
                 d_conv=4, expand=2, n_layers=2, dropout=0.1, T_seq=500):
        super().__init__()

        # 输入茎干: Conv1d 将少量频域特征升维至隐空间
        self.stem = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2)
        self.stem_norm = nn.LayerNorm(d_model)
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(dropout)

        # Mamba 堆叠
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, T_seq)
            for _ in range(n_layers)
        ])

        # 输出头
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()       # 确保 ŷ ∈ [0, 1]
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)                     # [B, F, T]
        x = self.stem(x)                             # [B, d_model, T]
        x = x.permute(0, 2, 1)                      # [B, T, d_model]
        x = self.stem_norm(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)

        for layer in self.layers:
            x = layer(x)                             # [B, T, d_model]

        x = self.head_norm(x)
        y = self.head(x)                             # [B, T, 1]
        return y


# ═══════════════════════════════════════════════════════════════
# 4. 基线模型 (用于消融实验)
# ═══════════════════════════════════════════════════════════════
class GRUBaseline(nn.Module):
    """GRU 基线: 与 FrequencyMamba 相同的茎干与头部, 仅时序骨干不同"""
    def __init__(self, input_dim=2, d_model=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2)
        self.stem_norm = nn.LayerNorm(d_model)
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                          batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = x.permute(0, 2, 1)
        x = self.stem_norm(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)
        x, _ = self.gru(x)
        x = self.head_norm(x)
        return self.head(x)


class LSTMBaseline(nn.Module):
    """LSTM 基线"""
    def __init__(self, input_dim=2, d_model=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2)
        self.stem_norm = nn.LayerNorm(d_model)
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = x.permute(0, 2, 1)
        x = self.stem_norm(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)
        x, _ = self.lstm(x)
        x = self.head_norm(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════
# 5. 评估指标
# ═══════════════════════════════════════════════════════════════
def evaluate(y_pred, y_true):
    """
    多维度评估回归质量。

    Returns:
        mse:      均方误差
        mae:      平均绝对误差
        trend_r:  Pearson r (趋势一致性 — 核心论文指标)
        smoothness: 预测曲线平滑度 (1 / 平均二阶差分绝对值)
    """
    y_pred = y_pred.detach()
    y_true = y_true.detach()
    mse = F.mse_loss(y_pred, y_true).item()
    mae = F.l1_loss(y_pred, y_true).item()

    # Pearson 相关系数 (衡量趋势一致性)
    pred_flat = y_pred.flatten()
    true_flat = y_true.flatten()
    pred_centered = pred_flat - pred_flat.mean()
    true_centered = true_flat - true_flat.mean()
    cov = (pred_centered * true_centered).sum()
    denom = (pred_centered.norm() * true_centered.norm() + 1e-8)
    pearson_r = (cov / denom).item()

    # 平滑度: 平均二阶差分 (越小越平滑)
    second_diff = torch.diff(y_pred.squeeze(-1), n=2, dim=1).abs().mean().item()
    smoothness = 1.0 / (second_diff + 1e-8)

    return {'mse': mse, 'mae': mae, 'pearson_r': pearson_r, 'smoothness': smoothness}


# ═══════════════════════════════════════════════════════════════
# 6. 训练器
# ═══════════════════════════════════════════════════════════════
def train_epoch(model, X, Y, optimizer, criterion, batch_size):
    """单 epoch 训练, 返回平均 loss"""
    model.train()
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)
    total_loss, n_batches = 0.0, 0

    for start in range(0, n_samples, batch_size):
        idx = indices[start:start + batch_size]
        x_batch = X[idx].to(DEVICE)
        y_batch = Y[idx].to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, X, Y):
    """全量验证"""
    model.eval()
    y_pred = model(X.to(DEVICE))
    return evaluate(y_pred.cpu(), Y)


# ═══════════════════════════════════════════════════════════════
# 7. 可视化引擎
# ═══════════════════════════════════════════════════════════════
def plot_results(X, Y, predictions, model_names, epoch, save_path):
    """
    论文级四联图:
      (a) 输入特征 (HER + Δθ) — 展示噪声水平
      (b) 各模型预测 vs 真实 TTE — 核心对比
      (c) 预测残差时序 — 误差分布诊断
      (d) 训练 Loss 曲线
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 选第一个受试者的数据做可视化
    t = np.arange(X.shape[1])

    # (a) 输入特征: 双 Y 轴
    ax = axes[0, 0]
    ax2 = ax.twinx()
    line1, = ax.plot(t, X[0, :, 0].numpy(), 'blue', alpha=0.6, linewidth=0.8, label='HER (noisy)')
    line2, = ax2.plot(t, X[0, :, 1].numpy(), 'red', alpha=0.6, linewidth=0.8, label='Δθ (°)')
    ax.set_xlabel('Time step (1Hz)')
    ax.set_ylabel('HER', color='blue')
    ax2.set_ylabel('Δθ (°)', color='red')
    ax.set_title('(a) Input: Noisy Frequency Features $X_t$')
    ax.legend([line1, line2], ['HER', 'Δθ'], loc='upper left')
    ax.grid(True, alpha=0.3)

    # (b) 预测 vs 真实 TTE
    ax = axes[0, 1]
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    ax.plot(t, Y[0, :, 0].numpy(), 'k-', linewidth=2.5, label='GT $Y_t$ (TTE)', zorder=10)
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        ax.plot(t, pred[0, :, 0].numpy(), colors[i % len(colors)],
                linewidth=1.8, alpha=0.85, label=f'{name}')
    ax.set_xlabel('Time step (1Hz)')
    ax.set_ylabel('Normalized TTE')
    ax.set_title(f'(b) Predicted vs Ground Truth TTE (Epoch {epoch})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (c) 残差
    ax = axes[1, 0]
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        residual = (pred[0, :, 0] - Y[0, :, 0]).numpy()
        ax.plot(t, residual, colors[i % len(colors)],
                linewidth=1.2, alpha=0.7, label=f'{name} residual')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time step (1Hz)')
    ax.set_ylabel('Residual (ŷ - y)')
    ax.set_title('(c) Prediction Residuals')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # (d) 留空 (Loss 曲线在训练过程中动态保存)
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Loss curves saved separately\n(see loss_curves.png)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color='gray')
    ax.set_title('(d) Training Dynamics')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"    → 图表保存至: {save_path}")


def plot_loss_curves(history, save_path):
    """训练 Loss 消融对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, logs in history.items():
        epochs = range(1, len(logs['train_loss']) + 1)
        ax1.plot(epochs, logs['train_loss'], linewidth=1.5, alpha=0.8, label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    for name, logs in history.items():
        epochs = range(1, len(logs['pearson_r']) + 1)
        ax2.plot(epochs, logs['pearson_r'], linewidth=1.5, alpha=0.8, label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pearson r')
    ax2.set_title('Trend Consistency (Pearson r)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0.7, 1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"    → Loss 曲线保存至: {save_path}")


def plot_ablation_summary(metrics, save_path):
    """消融实验终局对比柱状图"""
    model_names = list(metrics.keys())
    metric_names = ['mse', 'mae', 'pearson_r']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric in zip(axes, metric_names):
        values = [metrics[m][metric] for m in model_names]
        colors = ['tab:blue', 'tab:red', 'tab:green'][:len(model_names)]
        bars = ax.bar(model_names, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3, axis='y')
        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Ablation Study: Temporal Backbone Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"    → 消融柱状图保存至: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 8. 主入口: 合成数据沙盘 → 训练 → 消融
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  Frequency-Mamba: 合成数据沙盘与消融实验")
    print(f"  设备: {DEVICE}")
    print("=" * 64)

    # ---- 超参数 ----
    N_SUBJECTS = 6           # 模拟受试者数
    T = 500                  # 时间步 (500s @ 1Hz)
    F_IN = 2                 # 特征维度 (HER, Δθ)
    D_MODEL = 64             # 隐空间维度
    D_STATE = 16             # SSM 状态维度
    N_LAYERS = 2             # Mamba/GRU/LSTM 层数
    BATCH_SIZE = N_SUBJECTS  # 全量 batch: SSM 选择性机制需跨受试者梯度一致性
    EPOCHS = 200             # 完整训练
    LR = 2e-3
    WEIGHT_DECAY = 1e-4

    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUT_DIR, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    # ---- 合成数据 ----
    print("\n[1/4] 生成合成频域特征数据...")
    X, Y = generate_synthetic_cohort(n_subjects=N_SUBJECTS, T=T,
                                      noise_config='heteroscedastic')
    print(f"      X: {X.shape}, Y: {Y.shape}")

    # ---- 模型初始化 ----
    print("\n[2/4] 初始化模型: FrequencyMamba + GRU + LSTM...")
    models = {
        'FrequencyMamba': FrequencyMamba(
            input_dim=F_IN, d_model=D_MODEL, d_state=D_STATE,
            n_layers=N_LAYERS, T_seq=T
        ),
        'GRU': GRUBaseline(input_dim=F_IN, d_model=D_MODEL, n_layers=N_LAYERS),
        'LSTM': LSTMBaseline(input_dim=F_IN, d_model=D_MODEL, n_layers=N_LAYERS),
    }

    for name, model in models.items():
        model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      {name}: {n_params:,} 可训练参数")

    # ---- 训练 ----
    print(f"\n[3/4] 训练 {EPOCHS} epochs...")
    criterion = nn.MSELoss()
    history = {name: {'train_loss': [], 'pearson_r': [], 'mse': [], 'mae': []}
               for name in models}

    best_metrics = {}
    for name, model in models.items():
        print(f"\n  --- 训练 {name} ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        best_loss = float('inf')

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, X, Y, optimizer, criterion, BATCH_SIZE)
            scheduler.step()

            history[name]['train_loss'].append(train_loss)

            if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
                metrics = validate(model, X, Y)
                history[name]['pearson_r'].append(metrics['pearson_r'])
                history[name]['mse'].append(metrics['mse'])
                history[name]['mae'].append(metrics['mae'])

                status = (f"Epoch {epoch:3d}/{EPOCHS} | "
                          f"Loss: {train_loss:.6f} | "
                          f"MSE: {metrics['mse']:.6f} | "
                          f"Pearson r: {metrics['pearson_r']:.4f} | "
                          f"Smooth: {metrics['smoothness']:.2f}")
                if metrics['mse'] < best_loss:
                    best_loss = metrics['mse']
                    status += " ★"
                    torch.save(model.state_dict(),
                               os.path.join(run_dir, f'{name}_best.pt'))
                print(status)

        best_metrics[name] = validate(model, X, Y)

    # ---- 可视化与消融 ----
    print("\n[4/4] 生成论文级消融实验图表...")

    # 使用最佳模型权重做预测
    all_predictions = []
    model_names = []
    for name, model in models.items():
        best_path = os.path.join(run_dir, f'{name}_best.pt')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            pred = model(X.to(DEVICE)).cpu()
        all_predictions.append(pred)
        model_names.append(name)

    # 四联图
    plot_results(X, Y, all_predictions, model_names, EPOCHS,
                 os.path.join(run_dir, 'prediction_quad.png'))

    # Loss 曲线
    plot_loss_curves(history, os.path.join(run_dir, 'loss_curves.png'))

    # 消融柱状图
    plot_ablation_summary(best_metrics,
                          os.path.join(run_dir, 'ablation_bars.png'))

    # ---- 终端报告 ----
    print("\n" + "=" * 64)
    print("  消融实验终局报告 (Final Ablation Report)")
    print("=" * 64)
    header = f"{'Model':<20} {'MSE↓':>10} {'MAE↓':>10} {'Pearson r↑':>12} {'Smooth↑':>10}"
    print(header)
    print("-" * len(header))
    for name in models:
        m = best_metrics[name]
        print(f"{name:<20} {m['mse']:>10.6f} {m['mae']:>10.6f} "
              f"{m['pearson_r']:>12.4f} {m['smoothness']:>10.2f}")

    # 找出最优模型
    best_model = min(best_metrics, key=lambda k: best_metrics[k]['mse'])
    best_r = max(best_metrics, key=lambda k: best_metrics[k]['pearson_r'])
    print(f"\n  MSE 最优: {best_model} ({best_metrics[best_model]['mse']:.6f})")
    print(f"  趋势一致最优: {best_r} (Pearson r={best_metrics[best_r]['pearson_r']:.4f})")
    print(f"\n  完整输出目录: {run_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
