from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path


def get_transforms(train: bool = True):
    base = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    if train:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)
