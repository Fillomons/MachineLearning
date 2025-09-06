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


def main():
    cfg = load_config("configs/config.json", "configs/config_schema.json")
    device = torch.device(cfg["device"])

    _, _, test_loader, class_names = get_loaders(
        cfg["paths"]["processed_root"], cfg["training"]["batch_size"]
    )

    model = build_model(
        cfg["model"]["name"], len(class_names), cfg["model"]["finetune"]
    ).to(device)

    best_path = Path(cfg["paths"]["models_root"]) / f"{cfg['model']['name']}_best.pt"
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    ys, yhs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            yhat = logits.argmax(1).cpu()
            ys.extend(y.tolist())
            yhs.extend(yhat.tolist())

    acc = accuracy_score(ys, yhs)
    prec, rec, f1, _ = precision_recall_fscore_support(
        ys, yhs, average="macro", zero_division=0
    )
    print(
        {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
    )

    plot_confusion_matrix(
        ys,
        yhs,
        class_names,
        out_path=str(
            Path(cfg["paths"]["plots_root"]) / f"cm_{cfg['model']['name']}.png"
        ),
    )


if __name__ == "__main__":
    main()
