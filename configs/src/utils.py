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
