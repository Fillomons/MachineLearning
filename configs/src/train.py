from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.datasets import get_loaders
from src.models import (
    get_resnet18,
    CNN_Simple,
)
from src.utils import EarlyStopping, save_checkpoint


def build_model(name: str, num_classes: int, finetune: bool):
    if name == "cnn_simple":
        return CNN_Simple(num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes, finetune)
    raise ValueError(f"Unknown model name: {name}")


def main():
    cfg = load_config("configs/config.json", "configs/config_schema.json")
    device = torch.device(cfg["device"])

    train_loader, val_loader, _, class_names = get_loaders(
        cfg["paths"]["processed_root"], cfg["training"]["batch_size"]
    )

    model = build_model(
        cfg["model"]["name"], len(class_names), cfg["model"]["finetune"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    es = EarlyStopping(patience=cfg["training"]["patience"], mode="min")
    writer = SummaryWriter(cfg["paths"]["tb_root"]) if cfg["tensorboard"] else None

    best_val = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)

        if writer:
            writer.add_scalar("Loss/train", tr_loss, epoch)
            writer.add_scalar("Loss/val", va_loss, epoch)
            writer.add_scalar("Acc/train", tr_acc, epoch)
            writer.add_scalar("Acc/val", va_acc, epoch)
            writer.add_scalar("F1/train", tr_f1, epoch)
            writer.add_scalar("F1/val", va_f1, epoch)

        print(
            f"Epoch {epoch}: "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}"
        )

        # Save last
        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict()},
            Path(cfg["paths"]["models_root"]) / f"{cfg['model']['name']}_last.pt",
        )

        # Save best (by val loss)
        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict()},
                Path(cfg["paths"]["models_root"]) / f"{cfg['model']['name']}_best.pt",
            )

        # Early stopping
        es.step(va_loss)
        if es.should_stop:
            print("Early stopping triggered.")
            break

    if writer:
        writer.close()
