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


def get_loaders(processed_root: str, batch_size: int, num_workers: int = 2):
    root = Path(processed_root)

    train_ds = datasets.ImageFolder(root / "train", transform=get_transforms(True))
    val_ds = datasets.ImageFolder(root / "val", transform=get_transforms(False))
    test_ds = datasets.ImageFolder(root / "test", transform=get_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_ds.classes
