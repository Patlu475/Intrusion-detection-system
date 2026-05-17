# Comparative Intrusion Detection System

A comparative study of machine learning and deep learning models for network intrusion detection, implementing and evaluating **Random Forest**, **1D CNN**, and **Bidirectional LSTM with Attention** on two benchmark datasets.

Based on: Gamage & Samarabandu (2020), *Deep learning methods for network intrusion detection*, Journal of Network and Computer Applications, 169.

## Datasets

| Dataset | Classes | Features (after encoding) | Train Samples | Test Samples |
|---------|---------|--------------------------|---------------|--------------|
| **NSL-KDD** | 5 (Normal, DoS, Probe, R2L, U2R) | 122 | 125,973 | 22,544 |
| **UNSW-NB15** | 10 (Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms) | 196 | 175,341 | 82,332 |

## Models

- **Random Forest** — Baseline classical ML with GridSearchCV hyperparameter tuning
- **1D CNN** — Three convolutional blocks (64 → 128 → 256 filters) with batch normalization
- **BiLSTM + Attention** — Two-layer bidirectional LSTM [64, 32] with Bahdanau additive attention mechanism (extension beyond the original paper)

## Project Structure

```
├── data/
│   └── download.py            # Downloads both datasets
├── models/
│   ├── random_forest.py       # RF with GridSearchCV
│   ├── cnn.py                 # 1D CNN architecture
│   └── lstm.py                # BiLSTM + Attention architecture
├── utils/
│   ├── preprocess.py          # Data loading, encoding, scaling, SMOTE
│   └── evaluate.py            # Metrics, ROC, confusion matrices
├── train.py                   # Unified training CLI
├── compare.py                 # Model comparison on test set
├── report_figures.py          # Publication-quality figure generation
├── requirements.txt
└── IDS_Comparative_Study.ipynb # Google Colab notebook
```

## Quick Start (Google Colab — Recommended)

The fastest way to run this project is on Google Colab with a free T4 GPU:

1. Open `IDS_Comparative_Study.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Go to **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run All**

The notebook clones this repo, downloads datasets, trains all models on both datasets, and generates figures. Takes ~10–15 minutes on a T4 GPU.

Training is **checkpoint-resumable** — if the session disconnects, re-run the cells and training picks up from the last completed epoch.

## Local Setup

### Requirements

- Python 3.9+
- ~4 GB disk space (datasets + models)

### Installation

```bash
git clone https://github.com/Patlu475/Intrusion-detection-system.git
cd Intrusion-detection-system
pip install -r requirements.txt
```

### Download Datasets

```bash
python data/download.py --dataset all
```

Options: `--dataset nsl-kdd`, `--dataset unsw-nb15`, or `--dataset all`

### Training

Train all models on a dataset:

```bash
python train.py --dataset nsl-kdd --model all --epochs 50
python train.py --dataset unsw-nb15 --model all --epochs 50
```

Train a single model:

```bash
python train.py --dataset nsl-kdd --model rf          # Random Forest (with GridSearchCV)
python train.py --dataset nsl-kdd --model rf --quick   # Random Forest (default params, faster)
python train.py --dataset nsl-kdd --model cnn          # 1D CNN
python train.py --dataset nsl-kdd --model lstm         # BiLSTM + Attention
```

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs (CNN/LSTM) |
| `--batch-size` | 256 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--no-smote` | off | Disable SMOTE oversampling |
| `--quick` | off | Skip GridSearchCV for RF |
| `--seed` | 42 | Random seed |

Models are saved to `output/<dataset>/`. If training is interrupted, re-running the same command resumes from the last checkpoint. Already-completed models are skipped automatically.

### Compare Models

```bash
python compare.py --dataset nsl-kdd
python compare.py --dataset unsw-nb15
```

Prints per-class precision, recall, F1, and a side-by-side comparison table.

### Generate Figures

```bash
python report_figures.py --dataset nsl-kdd
python report_figures.py --dataset unsw-nb15
```

Generates the following in `figures/<dataset>/` (PNG + PDF):

- **Confusion matrices** — Normalized heatmaps with raw counts
- **ROC curves** — One-vs-rest per class + macro average
- **F1 comparison** — Grouped bar chart across all classes
- **Training curves** — Loss and macro F1 over epochs (CNN/LSTM)
- **Attention heatmap** — BiLSTM attention weights per attack class

## Preprocessing Pipeline

1. **Categorical encoding** — One-hot encoding of categorical features (3 for NSL-KDD, 3 for UNSW-NB15)
2. **Scaling** — StandardScaler fitted on training data only
3. **SMOTE** — Synthetic Minority Oversampling on training data only to handle class imbalance
4. **Class weights** — Computed from pre-SMOTE distribution and applied to CrossEntropyLoss

## Key Design Decisions

- **Macro F1** as primary metric (not accuracy) — the datasets are heavily imbalanced
- **SMOTE + class weights** — double defense against rare class underperformance
- **Test set as validation** — standard practice for NSL-KDD/UNSW-NB15 since train/test splits are fixed by the dataset authors
- **Early stopping** (patience=10) on validation macro F1 prevents overfitting
- **Attention mechanism** on BiLSTM — extends the original paper; enables interpretability of which feature groups the model focuses on per attack type

## Reference

Gamage, S., & Samarabandu, J. (2020). Deep learning methods in network intrusion detection: A survey and an objective comparison. *Journal of Network and Computer Applications*, 169, 102767.
