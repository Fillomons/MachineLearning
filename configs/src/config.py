import json
from jsonschema import validate
from pathlib import Path


def load_config(cfg_path: str, schema_path: str):
    cfg_path, schema_path = Path(cfg_path), Path(schema_path)

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    validate(instance=cfg, schema=schema)

    # create needed dirs
    for k in ["processed_root", "models_root", "plots_root", "tb_root"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)

    return cfg
