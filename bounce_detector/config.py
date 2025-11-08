from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any
import json

@dataclass
class DataConfig:
    train_glob: list[str] | None = None
    val_glob: list[str] | None = None
    test_glob: list[str] | None = None
    feature_fields: list[str] | None = None
    window_len: int = 200
    scaler_clip: float = 8.0
    use_valid_idx: bool = True
    label_name: str = "labels"
    seed: int = 42

@dataclass
class ModelConfig:
    in_features: int = 0
    window_len: int = 200
    hidden: int = 64
    k: int = 9
    layers: int = 2
    dropout: float = 0.1
    out_dim: int = 1

@dataclass
class TrainConfig:
    device: str = "auto"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    precision16: bool = True
    class_weight_pos: float | str = "auto"
    early_stopping_patience: int = 10

@dataclass
class EvalConfig:
    thresholds: list[float] | None = None
    metrics: list[str] | None = None

@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig | None = None
    version: str = "v1"
    git_commit: str | None = None

@dataclass
class RunScore:
    custom_score: float
    ones_score: float
    changes_score: float
    price_score: float
    auprc_score: float
    threshold: float
    thres_std_dev: float

def asdict_dc(dc: Any) -> dict:
    if is_dataclass(dc):
        return {k: asdict_dc(v) for k, v in asdict(dc).items()}
    if isinstance(dc, dict):
        return {k: asdict_dc(v) for k, v in dc.items()}
    if isinstance(dc, (list, tuple)):
        return [asdict_dc(v) for v in dc]
    return dc

def merge_dotset(base: dict, key: str, value: str):
    def parse_val(s: str):
        s = s.strip()
        try: return json.loads(s)
        except Exception: pass
        if s.lower() in ("true","false"): return s.lower() == "true"
        try:
            return float(s) if "." in s else int(s)
        except Exception:
            return s
    parts = key.split(".")
    cur = base
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = parse_val(value)

def load_config_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def resolve_run_config(cfg_dict: dict) -> RunConfig:
    data = cfg_dict.get("data", {})
    model = cfg_dict.get("model", {})
    train = cfg_dict.get("train", {})
    evalc = cfg_dict.get("eval", {})
    return RunConfig(
        data=DataConfig(**data),
        model=ModelConfig(**model),
        train=TrainConfig(**train),
        eval=EvalConfig(**evalc) if evalc else None,
        version=cfg_dict.get("version", "v1"),
        git_commit=cfg_dict.get("git_commit", None),
    )
