import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import load_config
from src.datasets import get_loaders
from src.models import CNN_Simple, get_resnet18
from src.utils import plot_confusion_matrix


def build_model(name: str, num_classes: int, finetune: bool):
    if name == "cnn_simple":
        return CNN_Simple(num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes, finetune)
    raise ValueError(f"Unknown model name: {name}")
