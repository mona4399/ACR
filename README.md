# Adaptive Confidence Regularization for Multimodal Failure Detection

<p align="center">
  <a href="https://www.linkedin.com/in/moru-liu-a124a5246">Moru Liu</a><sup>1</sup>
  · <a href="https://sites.google.com/view/dong-hao/">Hao Dong</a><sup>2</sup>
  · <!-- TODO add remaining co-authors with links and affiliations -->
</p>

<p align="center">
  <sup>1</sup>Technical University of Munich
  · <sup>2</sup>ETH Zurich
  · <!-- TODO Fraunhofer IKS / EPFL / etc. -->
</p>

<p align="center">
  • <a href="https://arxiv.org/abs/XXXX.XXXXX"><b>arXiv</b></a>
  • <a href="#citation"><b>BibTeX</b></a>
  •
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://huggingface.co/Mona4399"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-orange" alt="Hugging Face"></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="85%">
</p>

---

## Abstract

> _<!-- TODO: paste a 4-6 sentence abstract from the paper here. Keep it self-contained: name the problem (multimodal failure / misclassification detection), name the method (Adaptive Confidence Regularization, ACR), and state the main empirical claim (e.g. "consistent improvements in AURC and AUROC over MSP, Energy, and Doctor baselines across HMDB51, EPIC-Kitchens, and the HAC out-of-distribution setting"). -->_


## Updates

- **2026-XX-XX** — Initial code and checkpoints released.


## Installation

### Environment

We recommend creating a fresh conda environment. The code is tested with **Python 3.10**, **PyTorch 1.10**, **CUDA 11.3**, and **mmaction2 v0.13.0** (vendored under [`mmaction/`](mmaction/)).

```bash
conda create -n acr python=3.10 -y
conda activate acr

# Install PyTorch matching your CUDA version, e.g. for CUDA 11.3:
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# Install mmcv-full matching the PyTorch+CUDA build (use mim if you prefer):
pip install mmcv-full==1.2.7 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Project dependencies
pip install -r requirements.txt
```

A conda environment file is also provided in [`environment.yml`](environment.yml).

### Pretrained backbones

We use the SlowFast (R101) and SlowOnly (R50) backbones pretrained on Kinetics-400, both released by [mmaction2](https://github.com/open-mmlab/mmaction2). Download them once and place them under `HMDB-rgb-flow/pretrained_models/` and `EPIC-rgb-flow/pretrained_models/`:

```bash
# RGB backbone (SlowFast R101, ~265 MB)
wget https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth \
  -O HMDB-rgb-flow/pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth

# Flow backbone (SlowOnly R50, ~135 MB)
wget https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth \
  -O HMDB-rgb-flow/pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth

# (Symlink or copy the same two files into EPIC-rgb-flow/pretrained_models/)
```


## Datasets

We follow the data layout used in [MultiOOD](https://github.com/donghao51/MultiOOD) and [SimMMDG](https://github.com/donghao51/SimMMDG); if you have either of these prepared, ACR runs directly on the same folders.

### HMDB51 (+ Kinetics, UCF, HAC for OOD)

```
<DATAPATH>/
├── video/                       # raw HMDB51 .avi or .mp4 clips
└── flow/                        # pre-extracted TV-L1 flow as ..._flow_x.mp4, ..._flow_y.mp4
```

Splits for HMDB51 and Kinetics-600 are already included under `HMDB-rgb-flow/splits/`. For the HAC out-of-distribution benchmark, see the HAC release in [SimMMDG](https://github.com/donghao51/SimMMDG).

### EPIC-Kitchens (D3 — kitchen P22)

```
<DATAPATH>/
├── rgb/
│   ├── train/D3/<participant>/frame_xxxxxxxxxx.jpg
│   └── test/D3/<participant>/frame_xxxxxxxxxx.jpg
└── flow/
    ├── train/D3/<participant>/frame_xxxxxxxxxx.jpg
    └── test/D3/<participant>/frame_xxxxxxxxxx.jpg
```

Splits (`D3_train.pkl`, `D3_val.pkl`, `D3_test.pkl`) are included under `EPIC-rgb-flow/splits/`.

### Pre-extracted optical flow

We use TV-L1 optical flow extracted at the source frame rate, packaged as two MP4s per clip (one per flow component). The extraction procedure is described in the [MultiOOD data preparation guide](https://github.com/donghao51/MultiOOD#dataset-preparation); we do not re-distribute the flow files.


## Training

The headline launchers are in [`scripts/`](scripts/). Override the dataset path via the `DATAPATH` environment variable.

### HMDB51

```bash
DATAPATH=/path/to/HMDB51/ bash scripts/train_hmdb.sh
```

Equivalent direct invocation:

```bash
cd HMDB-rgb-flow
python train_video_flow.py \
    --dataset HMDB \
    --datapath /path/to/HMDB51/ \
    --lr 1e-4 --bsz 16 --nepochs 50 --num_workers 10 --seed 0 \
    --opt adam --use_single_pred --use_acl --acl_loss 2.0 \
    --save_best --save_checkpoint
```

### EPIC-Kitchens

```bash
DATAPATH=/path/to/EPIC-KITCHENS/ bash scripts/train_epic.sh
```

The checkpoint with the best validation accuracy is saved as `models/<log_name>_best.pt`.


## Evaluation

We support six confidence-score variants out of the box:
`msp`, `energy`, `max-logit`, `entropy`, `var`, `doctor` (see [Granese et al., NeurIPS 2021](https://arxiv.org/abs/2106.02395)).

### HMDB51

```bash
DATAPATH=/path/to/HMDB51/ \
CKPT=HMDB-rgb-flow/checkpoints/acr_hmdb_best.pt \
bash scripts/test_hmdb.sh
```

Equivalent direct invocation:

```bash
cd HMDB-rgb-flow
python test_video_flow.py \
    --dataset HMDB --datapath /path/to/HMDB51/ \
    --bsz 16 --num_workers 2 \
    --score msp \
    --resumef checkpoints/acr_hmdb_best.pt
```

Switch `--score` to `energy`, `max-logit`, `entropy`, `var`, or `doctor` to reproduce the other rows of the results table below.

### EPIC-Kitchens

```bash
DATAPATH=/path/to/EPIC-KITCHENS/ \
CKPT=EPIC-rgb-flow/checkpoints/acr_epic_best.pt \
bash scripts/test_epic.sh
```


## Released Checkpoints

Trained ACR checkpoints are released on Hugging Face: <https://huggingface.co/Mona4399/ACR>.

| Dataset           | Backbone       | Modalities | AURC ↓ | AUROC ↑ | FPR@95 ↓ | Acc ↑ | Download |
|-------------------|----------------|------------|--------|---------|----------|-------|----------|
| HMDB51            | SlowFast R101  | RGB + Flow | _XX.XX_ | _XX.XX_ | _XX.XX_ | _XX.XX_ | [acr_hmdb_best.pt](https://huggingface.co/Mona4399/ACR/blob/main/acr_hmdb_best.pt) |
| EPIC-Kitchens (D3)| SlowFast R101  | RGB + Flow | _XX.XX_ | _XX.XX_ | _XX.XX_ | _XX.XX_ | [acr_epic_best.pt](https://huggingface.co/Mona4399/ACR/blob/main/acr_epic_best.pt) |

<!-- TODO fill in the numbers from the paper / your latest runs. Reported scores use MSP unless noted. -->

> ⚠️ **Note on `--use_mfs`.** If you train with `--use_mfs`, the classifier head adds one extra class for the synthetic mixed feature. Checkpoints trained with `--use_mfs` therefore **must** be evaluated with `--use_mfs` as well; otherwise `load_state_dict` raises a shape mismatch.







## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Liu_2026_CVPR,
    author    = {Liu, Moru and Dong, Hao and Fink, Olga and Trapp, Mario},
    title     = {Adaptive Confidence Regularization for Multimodal Failure Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15850-15859}
}
```


## Related Projects

- [FeatureMixing](https://github.com/mona4399/FeatureMixing) — Extremely Simple Multimodal Outlier Synthesis for OOD Detection and Segmentation (NeurIPS 2025)
- [MultiOOD](https://github.com/donghao51/MultiOOD) — Scaling Out-of-Distribution Detection for Multiple Modalities
- [SimMMDG](https://github.com/donghao51/SimMMDG) — A Simple and Effective Framework for Multimodal Domain Generalization
- [Awesome-Multimodal-Adaptation](https://github.com/donghao51/Awesome-Multimodal-Adaptation) — Survey of multimodal adaptation and generalization


## Acknowledgements

This codebase builds on the excellent open-source work of [mmaction2](https://github.com/open-mmlab/mmaction2), [MultiOOD](https://github.com/donghao51/MultiOOD), and [SimMMDG](https://github.com/donghao51/SimMMDG). We thank the authors for making their code and data available.


