import os, random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=5, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False

    def step(self, metric):
        improve = (metric < self.best) if self.best is not None else True
        if self.mode == "max":
            improve = (metric > self.best) if self.best is not None else True

        if improve:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.num_bad >= self.patience:
            self.should_stop = True

        return improve


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
