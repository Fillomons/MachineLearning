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
