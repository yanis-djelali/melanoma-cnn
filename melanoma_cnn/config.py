"""Paths and constants shared by the scripts."""
from pathlib import Path

DATA_DIR = Path("melanoma-cancer-dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "test"  # the Kaggle "test" split is used as the validation set

RESULTS_DIR = Path("results")
HISTORY_DIR = RESULTS_DIR / "history"
FIGURES_DIR = Path("figures")
CHECKPOINT_DIR = Path("checkpoints")
TENSORBOARD_DIR = Path("runs")

# Per-channel statistics of the training set, measured with scripts/compute_stats.py
MEAN = [0.7228, 0.5555, 0.5390]
STD = [0.1886, 0.1974, 0.2110]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 4
SEED = 0
