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
