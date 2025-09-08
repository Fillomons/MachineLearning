from pathlib import Path
import pandas as pd
import shutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # <-- necessario

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

    files_per_class = {}
    for c in classes:
        cls_dir = raw_root / c
        assert cls_dir.exists(), f"Class folder missing: {cls_dir}"
        files = sorted([p for p in cls_dir.iterdir() if p.is_file()])
        files_per_class[c] = files

    report_rows = []
    for c, files in files_per_class.items():
        n_total = len(files)

        n_train = (
            budget["train"] if budget["train"] > 0 else int(n_total * split["train"])
        )
        n_val = budget["val"] if budget["val"] > 0 else int(n_total * split["val"])
        n_test = budget["test"] if budget["test"] > 0 else (n_total - n_train - n_val)

        n_train = max(0, min(n_train, n_total))
        remaining = n_total - n_train
        n_val = max(0, min(n_val, remaining))
        remaining -= n_val
        n_test = max(0, min(n_test, remaining))

        train_files, temp_files = train_test_split(
            files, train_size=n_train, shuffle=True, random_state=cfg["seed"]
        )

        if len(temp_files) > 0:
            if n_val > 0:
                val_files, rest_files = train_test_split(
                    temp_files, train_size=n_val, shuffle=True, random_state=cfg["seed"]
                )
            else:
                val_files, rest_files = [], temp_files

            if n_test > 0:
                test_files, _ = train_test_split(
                    rest_files,
                    train_size=n_test,
                    shuffle=True,
                    random_state=cfg["seed"],
                )
            else:
                test_files = []
        else:
            val_files, test_files = [], []

        copy_subset(train_files, proc_root / "train" / c, len(train_files))
        copy_subset(val_files, proc_root / "val" / c, len(val_files))
        copy_subset(test_files, proc_root / "test" / c, len(test_files))

        report_rows.append(
            {
                "class": c,
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
                "total": len(train_files) + len(val_files) + len(test_files),
            }
        )

    df = pd.DataFrame(report_rows)
    df.to_csv("plots/split_report.csv", index=False)

    ax = df.set_index("class")[["train", "val", "test"]].plot(
        kind="bar", figsize=(7, 4)
    )
    ax.set_title("Images per split and class")
    ax.set_ylabel("# images")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("plots/split_bars.png", dpi=150)
    print(df)


if __name__ == "__main__":
    main()
