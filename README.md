

## Table of Contents
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
  - [Train](#train)
  - [Resume Training](#resume-training)
  - [Evaluate / Inference](#evaluate--inference)
- [Experiment Naming & Outputs](#experiment-naming--outputs)
- [Figure Reproduction](#figure-reproduction)
- [Configuration & Reproducibility](#configuration--reproducibility)
- [Results (Placeholders)](#results-placeholders)
- [Open Science (Code/Data/Weights)](#open-science-codedataweights)


## Repository Structure
```
figure/                          # Figure scripts for efficiency/accuracy plots
  ├─ inference vs flops.py
  ├─ matplotlib(x2).py
  ├─ matplotlib(x3).py
  └─ params vs inference.py

experiments/                     # Per-experiment outputs (logs, ckpts, metrics)
  ├─ scpfa-x2-pfa/
  ├─ scpfa-x4-6/  scpfa-x4-12/  scpfa-x4-42/  scpfa-x4-48/  scpfa-x4-54/  scpfa-x4-60/
  ├─ scpfa-x4-no/  scpfa-x4-pfa/  scpfa-x4-eca/  scpfa-x4-cbam/
  └─ ...

train/scpfa/                     # Training & model code
  ├─ configs/                    # (Optional) YAML/JSON configs
  ├─ datas/
  │  ├─ benchmark.py             # Benchmark datasets (Set5/Set14/B100/Urban100/Manga109)
  │  ├─ div2k.py                 # DIV2K loader
  │  └─ utils.py
  ├─ models/
  │  ├─ attention_helper.py      # Parameter-free attention helpers
  │  ├─ scpfa_block.py           # Core building blocks (Shift / PFA)
  │  ├─ scpfa_network.py         # Network definition
  │  ├─ test.py                  # Evaluation/Inference entry
  │  └─ utils.py
  ├─ environment.yml             # Conda environment (recommended)
  ├─ requirements.txt            # pip requirements
  ├─ train.py                    # General training entry
  ├─ train_scpfa.py              # SCPFAN-specific entry (if used)
  └─ utils.py
```

> **Naming convention**:  
> `scpfa-x4-54` → scale ×4, a variant with depth/stages 54.  
> `-no` → without attention (ablation); `-pfa` → parameter‑free attention; `-eca`/`-cbam` → comparison baselines.

---

## Environment Setup
```bash
# Conda (recommended)
conda env create -f train/scpfa/environment.yml
conda activate scpfa

# Or pip
pip install -r train/scpfa/requirements.txt
```
> Python ≥ 3.8 is recommended. Install CUDA/cuDNN per your GPU driver.

---

## Datasets
- **Train**: DIV2K (`DIV2K_train_HR`, `DIV2K_valid_HR`)  
- **Test**: Set5 / Set14 / B100 / Urban100 / Manga109

Example layout:
```
/path/to/datasets/
├─ DIV2K/
│  ├─ DIV2K_train_HR
│  └─ DIV2K_valid_HR
└─ benchmarks/
   ├─ Set5
   ├─ Set14
   ├─ B100
   ├─ Urban100
   └─ Manga109
```

Configure paths via configs or CLI flags `--data_root` and `--bench_root`.

---

## Quick Start

### Train
```bash
cd train/scpfa

python train.py \
  --scale 4 \
  --exp scpfa-x4-54 \
  --data_root /path/to/DIV2K \
  --bench_root /path/to/benchmarks
```
> If using the dedicated entry:
```bash
python train_scpfa.py --scale 4 --exp scpfa-x4-pfa \
  --data_root /path/to/DIV2K --bench_root /path/to/benchmarks
```

### Resume Training
```bash
python train.py \
  --scale 4 \
  --exp scpfa-x4-54 \
  --resume ../experiments/scpfa-x4-54/checkpoints/last.pth
```

### Evaluate / Inference
```bash
cd train/scpfa

python models/test.py \
  --scale 4 \
  --ckpt ../experiments/scpfa-x4-54/checkpoints/last.pth \
  --save_dir ../experiments/scpfa-x4-54/results
```

---

## Experiment Naming & Outputs
- Each run is identified by `--exp <name>` and saved under `experiments/<name>/`:
  - `checkpoints/` — model weights (`best.pth`, `last.pth`)
  - `logs/` — training/eval logs (txt/csv/json)
  - `results/` — images & metrics on benchmarks
  - `config.yaml` — snapshot of hyperparameters
- Recommended metrics: **PSNR/SSIM (Y channel)**, and optionally **LPIPS**.
- Always record **Params / FLOPs / Latency** for fair efficiency comparison.

---

## Figure Reproduction
Run from the repo root (ensure required CSV/JSON summary files exist):
```bash
python "figure/inference vs flops.py" --log experiments/summary_efficiency.json --out figures/inf_vs_flops.png
python "figure/params vs inference.py" --log experiments/summary_efficiency.json --out figures/params_vs_inf.png

python "figure/matplotlib(x2).py" --log experiments/summary_x2.csv --out figures/x2_bar.png
python "figure/matplotlib(x3).py" --log experiments/summary_x3.csv --out figures/x3_bar.png
```

> If your log filenames differ, update the `--log` path or edit the scripts accordingly.

---

## Configuration & Reproducibility
- Put dataset paths, batch size, learning rate, training steps/epochs, and benchmark lists in `configs/*.yaml`, or pass by CLI.
- Fix random seeds; save `config.yaml` and logs per run; report hardware and runtime.
- Track large weights with **Git LFS**:
  ```bash
  git lfs install
  git lfs track "*.pth" "*.pt" "*.bin"
  git add .gitattributes && git commit -m "Enable Git LFS"
  ```

**Example minimal YAML (optional):**
```yaml
# train/scpfa/configs/x4_scpfa54.yaml
scale: 4
data_root: /path/to/DIV2K
bench_root: /path/to/benchmarks
train:
  batch_size: 16
  epochs: 1000
  lr: 2.0e-4
  seed: 42
model:
  name: scpfa_network
  variant: x4-54
eval:
  benchmarks: [Set5, Set14, B100, Urban100, Manga109]
  save_images: true
```

---


## Open Science (Code/Data/Weights)
- Archive **code, datasets, and pretrained weights** with a permanent DOI (e.g., **Zenodo** / **Figshare**).  


---


