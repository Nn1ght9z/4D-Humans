# 4D-Humans + Frequency-Mamba: Monocular Video to Continuous Biomechanical Assessment

[![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)](https://arxiv.org/pdf/2305.20091.pdf)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ex4gE5v1bPR3evfhtG7sDHxQGsWwNwby?usp=sharing)

This repository extends the **[4D-Humans](https://github.com/shubham-goel/4D-Humans)** / **[HMR 2.0](https://arxiv.org/abs/2305.20091)** framework with a custom **Frequency-Mamba** pipeline for continuous biomechanical assessment from monocular video — targeting fatigue estimation, running kinematics analysis, and real-world sports motion understanding.

> ⚠️ **Status**: Research exploration / engineering prototype. This project is **not** a peer-reviewed publication. It is the result of independent experimentation with pose estimation, State-Space Models (Mamba), and real-world sensor data. See [Limitations & Lessons](#limitations--lessons) for known issues.

---

## What This Repository Adds (Beyond Upstream 4D-Humans)

### Core Pipeline (custom)

| File | Purpose |
|------|---------|
| [`pipeline.py`](pipeline.py) | **Frequency-Mamba** neural network: STFT frequency-domain features → Conv1d encoder → Mamba SSM → continuous TTE (Time-To-Exhaustion) regression. Core model definition. |
| [`robust_pipeline_1hz.py`](robust_pipeline_1hz.py) | **Phase 2 production pipeline**: 1Hz video frame sampler → HMR2 3D joint extraction → sagittal-plane kinematic angle computation → HER (Heart-rate Estimated from kinematics proxy) curve → anomaly-aware smoothing → FrequencyMamba fatigue prediction. |
| [`train_real.py`](train_real.py) | **Leave-One-Out Cross-Validation (LOOCV)** training harness for real collected data (`aligned_datasets_v26/`). Compares FrequencyMamba vs GRU vs LSTM baselines. |
| [`verdict_plot.py`](verdict_plot.py) | Anatomical-baseline-aligned verdict plot generator: produces per-subject TTE vs HER degradation curves with drift-compensated baseline normalization. |
| [`batch_process_v26.py`](batch_process_v26.py) | Batch processing script for v26 aligned tensor datasets. |
| [`batch_fit_parser.py`](batch_fit_parser.py) | Batch fitting & result extraction for LOOCV outputs. |

### Key Dependencies

- **4D-Humans / HMR 2.0** — monocular 3D human mesh & pose estimation (ViTDet + SMPL)
- **Mamba SSM** — selective state-space model (O(N) complexity, input-dependent gating)
- **Detectron2** — object detection backbone for person detection

---

## Quick Start

### Installation

```bash
git clone https://github.com/Nn1ght9z/4D-Humans.git
cd 4D-Humans

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate 4D-humans

# Option B: Pip
conda create --name 4D-humans python=3.10
conda activate 4D-humans
pip install torch
pip install -e .[all]
```

### SMPL Model

Download the [SMPL neutral model](http://smplify.is.tue.mpg.de) (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) and place it in `./data/`. Registration required on the SMPL website.

### Run Upstream HMR2.0 Demo

```bash
# Run on images (replace with your image folder)
python demo.py \
    --img_folder /path/to/your/images \
    --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame

# Run tracking on video
python track.py video.source="/path/to/your/video.mp4"
```

### Run Frequency-Mamba Pipeline

```bash
# Phase 1: Train the model with LOOCV on aligned datasets
python train_real.py

# Phase 2: Run production pipeline at 1Hz on a video
python robust_pipeline_1hz.py
```

---

## Architecture Overview

```
Video (monocular, any frame rate)
  │
  ▼
[Pass A] HMR2.0 + ViTDet  ──→  SMPL 3D joints (per frame)
  │
  ▼
[Pass B] Sagittal Kinematics  ──→  Knee angle θ(t), Hip angle φ(t)
  │
  ▼
[Pass C] HER proxy (kinematic → physiological estimate)
  │
  ▼
[Pass D] STFT sliding window (time→frequency domain)
  │
  ▼
[Pass E] FrequencyMamba
           ├── Conv1d projection (N_freq_bins → d_model)
           ├── Mamba SSM blocks (×2)
           └── MLP head → ŷ(TTE) ∈ [0,1]
  │
  ▼
Output: Continuous fatigue curve + per-frame biomechanical features
```

---

## Limitations & Lessons

This project was developed as a hands-on exploration of monocular video-based biomechanical assessment. Several limitations were identified during development:

- **Data**: Real-world data was self-collected under variable conditions with limited sample size and imperfect sensor synchronization (video + IMU).
- **Benchmarks**: No established benchmark or mature baseline exists for continuous fatigue estimation from monocular video — making quantitative evaluation difficult.
- **Pipeline integration**: HMR2 inference + Mamba sequence modeling are loosely coupled; end-to-end optimization or gradient flow between stages was not achieved.
- **Scope**: Started without systematic literature review or clear problem definition, leading to engineering-heavy exploration with limited publishable scientific contributions.

These issues reflect poor initial planning rather than fundamental flaws in the approach. The project served as valuable research training: understanding the importance of problem framing, dataset quality, baseline selection, and literature grounding **before** engineering begins.

---

## Project Structure

```
4D-Humans/
├── hmr2/                    # HMR2.0 model (modified from upstream)
│   ├── models/              # HMR2, SMPL wrapper, losses, discriminator
│   ├── datasets/            # Data loaders, preprocessing
│   ├── configs/             # Model & training configs
│   └── utils/               # Geometry, rendering, misc utilities
├── demo.py                  # Upstream image demo
├── eval.py                  # Upstream evaluation
├── track.py                 # Upstream video tracking (PHALP-based)
├── train.py                 # Upstream training script
├── pipeline.py              # FrequencyMamba model definition ★
├── robust_pipeline_1hz.py   # Production pipeline @ 1Hz ★
├── train_real.py            # LOOCV training harness ★
├── verdict_plot.py          # Verdict plot generator ★
├── batch_process_v26.py     # Batch processor ★
├── batch_fit_parser.py      # Batch fit parser ★
├── aligned_datasets_v26/    # v26 aligned tensor datasets
├── fit_data/                # Fitting/calibration data
├── imu_csv/                 # Raw IMU sensor CSV data
└── environment.yml          # Conda environment
```

★ = custom additions beyond upstream 4D-Humans

---

## Acknowledgements

This repository is built on top of:

- **[4D-Humans](https://github.com/shubham-goel/4D-Humans)** / **[HMR 2.0](https://arxiv.org/abs/2305.20091)** — Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, Jitendra Malik
- [ProHMR](https://github.com/nkolot/ProHMR), [SPIN](https://github.com/nkolot/SPIN), [SMPLify-X](https://github.com/vchoutas/smplify-x), [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [Detectron2](https://github.com/facebookresearch/detectron2)
- [Mamba](https://github.com/state-spaces/mamba) — Selective State Space Models

Thanks to [StabilityAI](https://stability.ai/) for the compute grant that enabled the original 4D-Humans work.

## Citing Original 4D-Humans

If you use the upstream HMR2.0 code, please cite:

```bibtex
@inproceedings{goel2023humans,
    title={Humans in 4{D}: Reconstructing and Tracking Humans with Transformers},
    author={Goel, Shubham and Pavlakos, Georgios and Rajasegaran, Jathushan and Kanazawa, Angjoo and Malik, Jitendra},
    booktitle={CVPR},
    year={2023}
}
```

## License

This repository inherits the [MIT License](LICENSE.md) from the upstream 4D-Humans project. All custom additions (*.py files at the repository root, excluding upstream-modified files) are also MIT-licensed.
