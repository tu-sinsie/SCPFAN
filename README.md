

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
  - [Train](#train)
  - [Resume Training](#resume-training)
  - [Evaluate / Inference](#evaluate--inference)
- [Experiment Naming \& Outputs](#experiment-naming--outputs)
- [Figure Reproduction](#figure-reproduction)
- [Configuration \& Reproducibility](#configuration--reproducibility)
- [Open Science (Code)](#open-science-code)


## Repository Structure
```
figure/                          # Figure scripts for efficiency/accuracy plots
  ├─ inference vs flops.py
  ├─ matplotlib(x2).py
  ├─ matplotlib(x3).py
  └─ params vs inference.py

test/experiments/                     # Per-experiment outputs (logs, ckpts, metrics)
  ├─ scpfa-x2-pfa/
  ├─ scpfa-x4-6/  scpfa-x4-12/ 
  ├─ scpfa-x4-no/ scpfa-x4-eca/
  ├─ scpfa-x4-cbam/
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
  └─ utils.py
```

> **Naming convention**:  
> `scpfa-x4-6` → scale ×4, a variant with depth/stages 6.  
> `-no` → without attention (ablation); `-pfa` → parameter‑free attention; `-eca`/`-cbam` → comparison baselines.

---

## Environment Setup
```bash
# Conda (recommended)
conda env create -f train/scpfa/environment.yml
conda activate scpfa

# Or pip
pip install -r requirements.txt
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

Configure paths via configs file.

---

## Quick Start

### Train
```bash
python train.py --config train/scpfa/configs/scpfa_x4.yml
```

### Resume Training
```bash
python train.py --config train/scpfa/configs/scpfa_x4.yml \
  --resume ./experiments/scpfa-x4/checkpoints/last.pth
```

### Evaluate / Inference
```bash
python test.py  --config train/scpfa/configs/scpfa_x4.yml \
  --ckpt ./experiments/scpfa-x4/checkpoints/last.pth
```

---

## Experiment Naming & Outputs
- Each run is identified by `--exp <name>` and saved under `experiments/<name>/`:
  - `checkpoints/` — model weights (`best.pth`, `last.pth`)
  - `logs/` — training/eval logs (txt/csv/json)
  - `results/` — images & metrics on benchmarks
  - `config.yaml` — snapshot of hyperparameters

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
model: 'scpfa'
## parameters for plain
scale: 4
rgb_range: 255
colors: 3
m_scpfa: 36
c_scpfa: 180
n_share: 0
r_expand: 2
act_type: 'relu'
window_sizes: [4, 8, 16]
pretrain:
attention_type: "SimAM"

## parameters for model training
patch_size: 192
batch_size: 64
data_repeat: 80
data_augment: 1

epochs: 1000
lr: 0.0002
decays: [250, 400, 450, 475, 500]
gamma: 0.5
log_every: 100
test_every: 1
log_path: "./experiments"
log_name:

## hardware specification
gpu_ids: [0,1]
threads: 8

## data specification
data_path: '/home/psdz/project/SR_datasets'
eval_sets: ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
```

---


## Open Science (Code)

- Archive **code with permanent DOIs.

**Code DOI:** [10.5281/zenodo.16892699](https://doi.org/10.5281/zenodo.16892699)  


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16892699.svg)](https://doi.org/10.5281/zenodo.16892699)


---


