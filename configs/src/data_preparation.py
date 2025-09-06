from pathlib import Path
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import seed_all
from src.config import load_config


def copy_subset(files, dst_dir: Path, max_n: int):
    """Copia al massimo max_n file nella cartella di destinazione."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files[:max_n]:
        shutil.copy2(f, dst_dir / f.name)


def main():
    cfg = load_config("configs/config.json", "configs/config_schema.json")
    seed_all(cfg["seed"])

    raw_root = Path(cfg["paths"]["raw_root"])
    proc_root = Path(cfg["paths"]["processed_root"])
    classes = cfg["classes"]
    split = cfg["split"]
    budget = cfg["budget_per_class"]

    assert raw_root.exists(), f"Raw root not found: {raw_root}"
    proc_root.mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    # Collect files per class
    files_per_class = {}
    for c in classes:
        cls_dir = raw_root / c
        assert cls_dir.exists(), f"Class folder missing: {cls_dir}"
        files = sorted([p for p in cls_dir.iterdir() if p.is_file()])
        random.shuffle(files)
        files_per_class[c] = files

    # Build splits with per-class budgets
    report_rows = []
    for c, files in files_per_class.items():
        n_train = min(budget["train"], len(files))
        n_val = min(budget["val"], max(0, len(files) - n_train))
        n_test = min(budget["test"], max(0, len(files) - n_train - n_val))

        # Fallback to ratios if budgets exceed available
        if n_train + n_val + n_test == 0:
            n_total = len(files)
            n_train = int(n_total * split["train"])
            n_val = int(n_total * split["val"])
            n_test = n_total - n_train - n_val

        # Slices
        copy_subset(files, proc_root / "train" / c, n_train)
        copy_subset(files[n_train:], proc_root / "val" / c, n_val)
        copy_subset(files[n_train + n_val :], proc_root / "test" / c, n_test)

        report_rows.append(
            {
                "class": c,
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "total": n_train + n_val + n_test,
            }
        )
