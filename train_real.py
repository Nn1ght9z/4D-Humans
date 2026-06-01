"""
train_real.py — 严格 LOOCV 真实数据训练: 6 组 v26 对齐张量 → FrequencyMamba vs GRU vs LSTM
输出 out-of-fold 预测图、逐折指标与特征审计表。
"""
import csv, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/4D-Humans')
from pipeline import FrequencyMamba, GRUBaseline, LSTMBaseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/root/4D-Humans/aligned_datasets_v26'
OUT_DIR = '/root/4D-Humans/pipeline_outputs/train_real'
os.makedirs(OUT_DIR, exist_ok=True)

D_MODEL, D_STATE, N_LAYERS, DROPOUT = 64, 16, 2, 0.1
EPOCHS, LR, WD = 200, 2e-3, 1e-4


def load_real_data(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    X_list, Y_list, gct_list, subjects = [], [], [], []
    max_T = 0
    for f in files:
        d = np.load(os.path.join(data_dir, f))
        feat = d['features'][:, :2].astype(np.float32)
        tte = d['tte'].reshape(-1, 1).astype(np.float32)
        X_list.append(torch.tensor(feat, dtype=torch.float32))
        Y_list.append(torch.tensor(tte, dtype=torch.float32))
        gct_list.append(d['gct'].astype(np.float32))
        subjects.append(f.replace('.npz', ''))
        max_T = max(max_T, len(tte))
    return X_list, Y_list, gct_list, subjects, max_T


def pad_to_max(X_list, Y_list, max_T):
    X_padded, Y_padded, masks = [], [], []
    for x, y in zip(X_list, Y_list):
        T = x.shape[0]
        X_padded.append(F.pad(x, (0, 0, 0, max_T - T)))
        Y_padded.append(F.pad(y, (0, 0, 0, max_T - T)))
        masks.append(torch.cat([torch.ones(T), torch.zeros(max_T - T)]))
    return torch.stack(X_padded), torch.stack(Y_padded), torch.stack(masks)


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def write_feature_audit(data_dir, save_path):
    rows = []
    for f in sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')]):
        d = np.load(os.path.join(data_dir, f))
        her = d['features'][:, 0]
        phase = d['features'][:, 1]
        tte = d['tte']
        gct = d['gct']
        vo = d['vo']
        n = len(tte)
        k = max(1, n // 5)
        her_q = np.nanpercentile(her, [0, 5, 50, 95, 100])
        phase_q = np.nanpercentile(phase, [0, 5, 50, 95, 100])
        phase_jumps = np.abs(np.diff(phase)) if n > 1 else np.array([])
        rows.append({
            'subject': f.replace('.npz', ''),
            'frames': n,
            'tte_start': float(tte[0]),
            'tte_end': float(tte[-1]),
            'her_min': float(her_q[0]),
            'her_p5': float(her_q[1]),
            'her_median': float(her_q[2]),
            'her_p95': float(her_q[3]),
            'her_max': float(her_q[4]),
            'phase_min': float(phase_q[0]),
            'phase_p5': float(phase_q[1]),
            'phase_median': float(phase_q[2]),
            'phase_p95': float(phase_q[3]),
            'phase_max': float(phase_q[4]),
            'phase_jumps_gt_180': int(np.sum(phase_jumps > 180.0)),
            'phase_jump_max': float(np.max(phase_jumps)) if len(phase_jumps) else 0.0,
            'her_tte_r': safe_corr(her, tte),
            'gct_tte_r': safe_corr(gct, tte),
            'vo_tte_r': safe_corr(vo, tte),
            'her_late_early_ratio': float(np.nanmedian(her[-k:]) / (np.nanmedian(her[:k]) + 1e-8)),
            'gct_late_early_delta': float(np.nanmedian(gct[-k:]) - np.nanmedian(gct[:k])),
        })
    if rows:
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f'  Feature audit → {save_path}')
    return rows


def masked_mse(y_pred, y_true, mask):
    diff = (y_pred - y_true) ** 2
    diff = diff * mask.unsqueeze(-1)
    return diff.sum() / (mask.sum() + 1e-8)


def train_epoch(model, X, Y, mask, opt, batch_size):
    model.train()
    idx = torch.randperm(X.shape[0])
    total_loss, n_batches = 0.0, 0
    for start in range(0, X.shape[0], batch_size):
        i = idx[start:start + batch_size]
        xb, yb, mb = X[i].to(DEVICE), Y[i].to(DEVICE), mask[i].to(DEVICE)
        opt.zero_grad()
        loss = masked_mse(model(xb), yb, mb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def predict_one(model, x):
    model.eval()
    pred = model(x.unsqueeze(0).to(DEVICE)).cpu()[0]
    return pred[:x.shape[0]]


def eval_series(pred, target):
    yp = pred[:, 0]
    yt = target[:, 0]
    mse = F.mse_loss(yp, yt).item()
    mae = F.l1_loss(yp, yt).item()
    pc = yp - yp.mean()
    tc = yt - yt.mean()
    r = ((pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)).item()
    d2 = torch.diff(yp, n=2).abs().mean().item() if len(yp) >= 3 else float('nan')
    return {'mse': mse, 'mae': mae, 'pearson_r': r, 'smoothness': 1.0 / (d2 + 1e-8)}


def make_models(max_T):
    return {
        'FrequencyMamba': FrequencyMamba(input_dim=2, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS, dropout=DROPOUT, T_seq=max_T),
        'GRU': GRUBaseline(input_dim=2, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT),
        'LSTM': LSTMBaseline(input_dim=2, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT),
    }


def fit_model(model, X_train, Y_train, mask_train, epochs, batch_size, log_prefix):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, X_train, Y_train, mask_train, opt, batch_size)
        sched.step()
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f'    {log_prefix} Ep {ep:3d}/{epochs} | train_loss={loss:.6f}')
    return model


def run_loocv(X_list, Y_list, subjects, max_T):
    rows = []
    oof_predictions = {name: [None] * len(subjects) for name in make_models(max_T)}

    for test_idx, held_out in enumerate(subjects):
        train_X = [x for i, x in enumerate(X_list) if i != test_idx]
        train_Y = [y for i, y in enumerate(Y_list) if i != test_idx]
        X_train, Y_train, mask_train = pad_to_max(train_X, train_Y, max_T)
        print(f'\n[LOOCV {test_idx + 1}/{len(subjects)}] Test={held_out}, Train={len(train_X)} subjects')

        for name, model in make_models(max_T).items():
            fit_model(model, X_train, Y_train, mask_train, EPOCHS, batch_size=len(train_X), log_prefix=name)
            pred = predict_one(model, X_list[test_idx])
            metrics = eval_series(pred, Y_list[test_idx])
            rows.append({'held_out': held_out, 'model': name, **metrics})
            oof_predictions[name][test_idx] = pred
            print(f'    TEST {name:<16} MSE={metrics["mse"]:.6f} MAE={metrics["mae"]:.6f} r={metrics["pearson_r"]:.4f}')

    return rows, oof_predictions


def summarize_loocv(rows):
    summary = {}
    for model in sorted(set(r['model'] for r in rows)):
        summary[model] = {}
        vals = [r for r in rows if r['model'] == model]
        for key in ['mse', 'mae', 'pearson_r', 'smoothness']:
            arr = np.array([v[key] for v in vals], dtype=np.float64)
            summary[model][f'{key}_mean'] = float(np.nanmean(arr))
            summary[model][f'{key}_std'] = float(np.nanstd(arr))
    return summary


def save_loocv(rows, summary):
    metrics_path = os.path.join(OUT_DIR, 'loso_metrics.csv')
    summary_path = os.path.join(OUT_DIR, 'loso_summary.json')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['held_out', 'model', 'mse', 'mae', 'pearson_r', 'smoothness'])
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  LOOCV metrics → {metrics_path}')
    print(f'  LOOCV summary → {summary_path}')


def plot_oof_killer_graph(X_list, Y_list, oof_predictions, gct_list, subjects, save_path):
    n_subjects = len(subjects)
    n_cols = min(3, max(1, n_subjects))
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    colors = {'FrequencyMamba': 'tab:orange', 'GRU': 'tab:green', 'LSTM': 'tab:purple'}

    for s, subject in enumerate(subjects):
        ax = axes[s]
        ax2 = ax.twinx()
        T = len(Y_list[s])
        t = np.arange(T)
        her = X_list[s][:, 0].numpy()
        tte = Y_list[s][:, 0].numpy()
        gct = gct_list[s]

        l1, = ax.plot(t, her, 'blue', alpha=0.45, linewidth=0.8, label='HER (clipped)')
        l2, = ax.plot(t, tte, 'k-', linewidth=1.6, label='TTE (GT)')
        pred_lines = []
        for name, preds in oof_predictions.items():
            pred = preds[s][:T, 0].numpy()
            line, = ax.plot(t, pred, color=colors[name], linewidth=1.2, alpha=0.85, label=f'{name} OOF')
            pred_lines.append(line)
        l3, = ax2.plot(t, gct[:T], 'red', linestyle='--', linewidth=1.1, alpha=0.75, label='GCT (ms)')

        ax.set_xlabel('Time step (1Hz)')
        ax.set_ylabel('HER / TTE', color='blue')
        ax2.set_ylabel('GCT (ms)', color='red')
        ax.set_title(subject)
        ax.grid(True, alpha=0.3)
        lines = [l1, l2] + pred_lines + [l3]
        ax2.legend(lines, [x.get_label() for x in lines], loc='upper left', fontsize=7)

    for ax in axes[n_subjects:]:
        ax.set_visible(False)

    fig.suptitle('Out-of-Fold Killer Graph: Clipped HER vs GCT with LOOCV Predictions', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'  OOF Killer Graph → {save_path}')


def print_summary_table(summary):
    print('\n' + '=' * 72)
    print('  Strict LOOCV Final Report (Mean ± Std over held-out subjects)')
    print('=' * 72)
    print(f'{"Model":<18} {"MSE↓":>18} {"MAE↓":>18} {"Pearson r↑":>18}')
    print('-' * 72)
    for model in ['FrequencyMamba', 'GRU', 'LSTM']:
        m = summary[model]
        print(f'{model:<18} '
              f'{m["mse_mean"]:.6f}±{m["mse_std"]:.6f}   '
              f'{m["mae_mean"]:.6f}±{m["mae_std"]:.6f}   '
              f'{m["pearson_r_mean"]:.4f}±{m["pearson_r_std"]:.4f}')
    print('=' * 72)


def main():
    print('=' * 64)
    print(f'  Strict Real-Data LOOCV: Device={DEVICE}')
    print('=' * 64)

    X_list, Y_list, gct_list, subjects, max_T = load_real_data(DATA_DIR)
    print(f'\n[1/4] Loaded {len(subjects)} subjects, max_T={max_T}')
    for s, x, y in zip(subjects, X_list, Y_list):
        print(f'  {s}: T={x.shape[0]}, HER∈[{x[:,0].min():.2f},{x[:,0].max():.2f}], TTE∈[{y.min():.2f},{y.max():.2f}]')

    print('\n[2/4] Feature audit')
    write_feature_audit(DATA_DIR, os.path.join(OUT_DIR, 'feature_audit.csv'))

    print('\n[3/4] Strict leave-one-subject-out training')
    rows, oof_predictions = run_loocv(X_list, Y_list, subjects, max_T)
    summary = summarize_loocv(rows)
    save_loocv(rows, summary)

    print('\n[4/4] Plot out-of-fold killer graph')
    plot_oof_killer_graph(X_list, Y_list, oof_predictions, gct_list, subjects,
                          os.path.join(OUT_DIR, 'killer_graph_real.png'))
    print_summary_table(summary)
    print(f'  输出: {OUT_DIR}/')


if __name__ == '__main__':
    main()
