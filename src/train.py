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

    # Resume from checkpoint
    resume_cfg = cfg["training"].get("resume", False)
    default_resume = (
        Path(cfg["paths"]["models_root"]) / f"{cfg['model']['name']}_last.pt"
    )
    resume_path = Path(cfg["training"].get("resume_path", default_resume))

    if resume_cfg and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
            print(f"[INFO] Resumed weights from: {resume_path}")
        else:
            print(f"[WARN] Checkpoint found but missing 'model_state': {resume_path}")
    elif resume_cfg:
        print(f"[WARN] Resume enabled but checkpoint not found: {resume_path}")

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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    import torch
    import numpy as np

    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    epoch_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    from sklearn.metrics import f1_score

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return epoch_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    import torch

    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    epoch_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    from sklearn.metrics import f1_score

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return epoch_loss, acc, f1


if __name__ == "__main__":
    main()
