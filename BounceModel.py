"""
Bounce Detector — L1 top-of-book time-series classifier (small-data friendly)

What you get in this single file:
- Loader for your .npz files (keys: sampled, labels, valid_idx, horizon)
- Window builder that matches your make_windows logic (no averaging)
- Minimal PyTorch model (tiny Temporal CNN → head)
- Train / Validate / Test helpers
- Save/Load interfaces and an inference wrapper usable from another script
- CLI: train → save → evaluate

Assumptions
- Binary classification (bounce starts = 1, else 0)
- Features come from sampled[feature_name] stacked into [T, F]
- Single stream assumed for now (S=1). You can extend to multi-stream later by stacking streams into F or adding a stream axis.

Usage
------
Train + Save + Evaluate
    python bouncedetector.py \
      --train data/train_*.npz \
      --val   data/val_*.npz \
      --test  data/test_*.npz \
      --features bid,ask,bid_size,ask_size \
      --out models/bounce.ckpt

Load elsewhere
--------------
    from bouncedetector import load_trained
    bd = load_trained('models/bounce.ckpt', device='cuda')
    # window_np is a NumPy array shaped [T, S=1, F] or [B, T, 1, F]
    p = bd.predict_window(window_np)

"""
from __future__ import annotations
import os
import glob
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

from pathlib import Path
import pickle, json, time
from copy import deepcopy


_REQUIRED_KEYS = {
    "model_class", "model_state", "model_init_kwargs",
    "feature_fields", "window_len", "scaler"
}

# ----------------------------
# Config (JSON-based; single source of truth)
# ----------------------------
from dataclasses import dataclass, asdict
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
    device: str = "auto"               # "cuda" | "cpu" | "auto"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    precision16: bool = True           # use autocast if GPU
    class_weight_pos: float | str = "auto"  # float or "auto"
    early_stopping_patience: int = 10

@dataclass
class EvalConfig:
    thresholds: list[float] | None = None   # optional sweep; if None use 0.5
    metrics: list[str] | None = None        # e.g., ["auprc","roc_auc","f1","accuracy"]

@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig | None = None
    version: str = "v1"
    git_commit: str | None = None

def _asdict_dc(dc):
    from dataclasses import is_dataclass
    if is_dataclass(dc):
        return {k: _asdict_dc(v) for k, v in asdict(dc).items()}
    if isinstance(dc, dict):
        return {k: _asdict_dc(v) for k, v in dc.items()}
    if isinstance(dc, (list, tuple)):
        return [_asdict_dc(v) for v in dc]
    return dc

def _merge_dotset(base: dict, key: str, value: str):
    """Apply --set key.sub=value into nested dict."""
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

def _load_config_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _resolve_run_config(cfg_dict: dict) -> RunConfig:
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

def _expand_globs(paths_or_globs: list[str] | None) -> list[str]:
    import glob
    if not paths_or_globs:
        return []
    out = []
    for p in paths_or_globs:
        out.extend(sorted(glob.glob(p)))
    return out or (paths_or_globs or [])

# ----------------------------
# Dataset utils (NPZ → windows)
# ----------------------------

def _load_one_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        sampled = npz["sampled"]
        labels = npz["labels"]
        valid_idx = npz["valid_idx"]
    return dict(sampled=sampled, labels=labels, valid_idx=valid_idx)


def make_windows_from_npz(npz_paths: List[str], feature_fields: List[str], window_len: int, use_valid_idx: bool = True) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """
    Build X, y from one or more .npz files using user's valid_idx.
    - X shape: [N, T, S=1, F]
    - y shape: [N]
    Returns (X, y, window_len, feature_fields)
    """
    X_list, Y_list = [], []
    window_len_global = None

    for p in npz_paths:
        data = _load_one_npz(p)
        sampled = data["sampled"]
        labels = data["labels"]
        valid_idx = data["valid_idx"].astype(int)

        # features: [T, F]
        feats = np.stack([sampled[name] for name in feature_fields], axis=1)
        T_total = feats.shape[0]


        if use_valid_idx:
            idx_iter = valid_idx
        else:
            idx_iter = range(window_len - 1, T_total)  # slide everywhere

        for t0 in idx_iter:
            t0 = int(t0)
            if t0 < window_len - 1 or t0 >= T_total:
                continue
            X_list.append(feats[t0 - window_len + 1 : t0 + 1][None, ...])  # [1, T, F]
            Y_list.append(labels[t0])

    if not X_list:
        print(p)
        raise("No training windows were constructed. Check valid_idx and window_len in your .npz files.")

    X = np.concatenate(X_list, axis=0)  # [N, T, F]
    X = X[:, :, None, :]                # [N, T, S=1, F]
    y = np.asarray(Y_list).astype(np.int64)
    return X, y, window_len, feature_fields


class WindowsDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()  # [T,1,F]
        if self.y is None:
            return x
        return x, torch.tensor(self.y[i], dtype=torch.long)

# ----------------------------
# NaN handling & normalization helpers
# ----------------------------

def _ffill_bfill_2d(X: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill over time for a 2D array [T, C].
    Preserves finite values, fills NaNs/inf along the time axis.
    """
    X = X.copy()
    T, C = X.shape
    # treat non-finite as missing
    mask = np.isfinite(X)
    # forward fill indices
    idx = np.where(mask, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    X_ff = X.copy()
    # fancy indexing for missing points
    miss = ~mask
    if miss.any():
        rows, cols = np.nonzero(miss)
        X_ff[rows, cols] = X_ff[idx[rows, cols], cols]
    # back fill for leading NaNs
    mask2 = np.isfinite(X_ff)
    idx2 = np.where(mask2, np.arange(T)[:, None], T - 1)
    np.minimum.accumulate(idx2[::-1, :], axis=0, out=idx2[::-1, :])
    X_fb = X_ff.copy()
    miss2 = ~mask2
    if miss2.any():
        rows, cols = np.nonzero(miss2)
        X_fb[rows, cols] = X_fb[idx2[rows, cols], cols]
    return X_fb


def ffill_bfill_time_batch(X: np.ndarray) -> np.ndarray:
    """Apply forward/back fill along time for each sample independently.
    Accepts X shaped [N, T, S, F] or [N, T, F]. Returns same shape.
    """
    orig_shape = X.shape
    if X.ndim == 4:
        N, T, S, F = X.shape
        Xr = X.reshape(N, T, S * F)
    elif X.ndim == 3:
        N, T, F = X.shape
        S = 1
        Xr = X
    else:
        raise ValueError(f"Unexpected shape for ffill_bfill_time_batch: {X.shape}")
    out = np.empty_like(Xr)
    for n in range(N):
        out[n] = _ffill_bfill_2d(Xr[n])
    if X.ndim == 4:
        out = out.reshape(N, T, S, F)
    return out


def fit_scaler_nan_safe(X_train: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-feature mean/std ignoring NaNs on TRAIN only.
    X_train: [N, T, S, F]
    Returns dict with arrays shaped [S, F]. If S=1 it's [1, F].
    """
    # Collapse N and T axes; keep S and F
    if X_train.ndim != 4:
        raise ValueError("Expected X_train with shape [N,T,S,F]")
    N, T, S, F = X_train.shape
    Xc = X_train.reshape(N * T, S, F)
    mean = np.nanmean(Xc, axis=0)
    std = np.nanstd(Xc, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}


def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip: float = 8.0) -> np.ndarray:
    """Apply mean/std normalization and sanitize non-finite values.
    Handles X of shape [N,T,S,F] or [T,S,F].
    """
    mean, std = scaler["mean"], scaler["std"]  # [S, F]
    if X.ndim == 4:
        # broadcast over N,T
        Xn = (X - mean) / (std + 1e-9)
    elif X.ndim == 3:
        Xn = (X - mean) / (std + 1e-9)
    else:
        raise ValueError(f"Unexpected shape for apply_scaler: {X.shape}")
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        Xn = np.clip(Xn, -clip, clip)
    return Xn

# ----------------------------
# Model — small Temporal CNN per stream (S=1)
# ----------------------------
class TinyTemporalCNN(nn.Module):
    """
    Depthwise-temporal CNN over features.
    Expects input [B, T, S=1, F] and flattens S into channel afterward.
    """
    def __init__(self, F: int, T: int, hidden: int = 64, n_out: int = 1, k: int = 9, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.F, self.T = F, T
        self.hidden = hidden
        self.k = k
        self.layers = layers
        self.dropout = dropout
        width = hidden
        mods = []
        in_ch = F
        for _ in range(layers):
            mods += [
                nn.Conv1d(in_ch, in_ch, k, padding=k//2, groups=in_ch),
                nn.Conv1d(in_ch, width, 1),
                nn.GELU(),
                nn.BatchNorm1d(width),
                nn.Dropout(dropout),
            ]
            in_ch = width
        self.tcn = nn.Sequential(*mods)
        self.head = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_out)
        )

    def forward(self, x):  # x: [B,T,S=1,F]
        B,T,S,F = x.shape
        x = x.view(B, T, F)          # drop S=1
        x = x.transpose(1, 2)        # [B, F, T]
        h = self.tcn(x)              # [B, width, T]
        h_last = h[:, :, -1]         # last time step summary
        return self.head(h_last)     # [B, n_out]

# ----------------------------
# Train / Eval
# ----------------------------

def _device_from_cfg(cfg: TrainConfig) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device
import numpy as np

def batched_loader(
    X,
    y=None,
    batch_size: int = 256,
    shuffle: bool = False,
    drop_last: bool = False,
):
    """
    Simple memory-light batch generator.

    Args:
        X: NumPy array or torch Tensor of shape [N, ...].
        y: Optional labels array/tensor of shape [N] or [N, ...].
        batch_size: Batch size.
        shuffle: Shuffle samples each epoch.
        drop_last: If True, drop the final smaller-than-batch batch.

    Yields:
        (xb, yb) for each batch. If y is None, yields (xb, None).
    """
    # Get length N (works for numpy arrays and torch tensors)
    try:
        N = len(X)
    except TypeError:
        raise ValueError("X must be indexable with a defined length.")

    if y is not None and len(y) != N:
        raise ValueError(f"Length mismatch: len(X)={N} but len(y)={len(y)}")

    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    # Determine the range respecting drop_last
    end = (N // batch_size) * batch_size if drop_last else N
    for start in range(0, end, batch_size):
        stop = min(start + batch_size, N)
        sel = idx[start:stop]

        # Support numpy or torch without forcing conversion
        xb = X[sel]
        yb = None if y is None else y[sel]

        yield xb, yb


def _filter_valid_and_binarize(y_true, y_prob, threshold=0.5):
    """
    - drop y == -1
    - y must be in {0,1} for metrics; if labels are {0} only, we still return sane metrics.
    """
    mask = (y_true >= 0) & np.isfinite(y_prob)
    y = y_true[mask]
    p = y_prob[mask]
    if y.size == 0:
        # nothing to evaluate
        return y, p, np.array([], dtype=int)

    # binarize probabilities
    yhat = (p >= threshold).astype(int)

    # ensure labels are 0/1; if dataset used {0} only, it's still fine.
    uniq = np.unique(y)
    # If someone used {-1,0} upstream and -1 was filtered, uniq might be {0} only.
    # If someone used {0,1}, we’re good.
    # If someone used {-1,1} (rare), after filtering uniq is {1}; map 1->1 already OK.
    return y, p, yhat

def _safe_metrics(y, p, yhat):
    """Compute metrics without crashing when there are no positives."""
    out = {}
    if y.size == 0:
        # No samples
        out.update(dict(n=0, acc=float("nan"), f1=float("nan"),
                        roc_auc=float("nan"), aucpr=float("nan"),
                        pos_rate=float("nan")))
        return out

    out["n"] = int(y.size)
    out["acc"] = float(accuracy_score(y, yhat))

    # F1 / ROC / PR need both classes present to be meaningful.
    uniq = np.unique(y)
    has_pos = np.any(y == 1)
    has_neg = np.any(y == 0)

    if has_pos and has_neg:
        out["f1"] = float(f1_score(y, yhat, zero_division=0))
        # Guard ROC AUC for degenerate probs
        try:
            out["roc_auc"] = float(roc_auc_score(y, p))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["aucpr"] = float(average_precision_score(y, p))
        except Exception:
            out["aucpr"] = float("nan")
    else:
        # Degenerate test fold: only one class present
        out["f1"] = float("nan")
        out["roc_auc"] = float("nan")
        out["aucpr"]  = float("nan")

    out["pos_rate"] = float(np.mean(y))
    return out


def train_from_npz(
    train_paths: list[str],
    val_paths: list[str],
    feature_fields: list[str],
    window_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    class_weight: float | str,
    precision16: bool,
    model: nn.Module,                     # ← pass the already-constructed model
    scaler_clip: float = 8.0,             # ← from config.data.scaler_clip
    early_stopping_patience: int = 10,    # ← from config.train.early_stopping_patience
) -> dict:
    """Trains and returns a checkpoint payload that is self-contained."""
    torch.manual_seed(42)  # or pass rc.data.seed
    device = torch.device(device)
    model.to(device)

    # ----- Prepare data (fit scaler on train only) -----
    Xtrain, ytrain, T_win, feat_list = make_windows_from_npz(
        train_paths, feature_fields, window_len=window_len, use_valid_idx=True
    )
    Xval, yval, _, _ = make_windows_from_npz(
        val_paths, feature_fields, window_len=window_len, use_valid_idx=True
    )

    scaler = fit_scaler_nan_safe(Xtrain)
    Xtrain = apply_scaler(Xtrain, scaler, clip=scaler_clip)
    Xval   = apply_scaler(Xval,   scaler, clip=scaler_clip)

    # ----- Class weight auto -----
    cw = class_weight
    if isinstance(cw, str) and cw == "auto":
        y_flat = ytrain[ytrain >= 0]
        pos = max(1, int((y_flat == 1).sum()))
        neg = max(1, int((y_flat == 0).sum()))
        cw = float(neg / pos)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = early_stopping_patience
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in batched_loader(Xtrain, ytrain, batch_size):  # your existing batching helper
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)
            optim.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(precision16 and device.type == "cuda")):
                logits = model(xb)         # expects [B, T, S=1, F] → adjust if needed
                loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        # ---- simple val loss for early-stop (kept minimal for now) ----
        model.eval()
        with torch.no_grad():
            vloss_sum, n = 0.0, 0
            for xb, yb in batched_loader(Xval, yval, batch_size):
                xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
                yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)
                logits = model(xb)
                vloss_sum += float(criterion(logits, yb).item()) * len(xb)
                n += len(xb)
            vloss = vloss_sum / max(1, n)

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Build self-contained payload -----
    F = Xtrain.shape[-1]
    T = window_len
    model_init_kwargs = dict(
        F=getattr(model, "F", F),
        T=getattr(model, "T", T),
        hidden=int(getattr(model, "hidden", 64)),
        n_out=1,
        k=int(getattr(model, "k", 9)),
        layers=int(getattr(model, "layers", 2)),
        dropout=float(getattr(model, "dropout", 0.1)),
    )

    payload = {
        "model_class": "TinyTemporalCNN",
        "model_state": model.state_dict(),
        "model_init_kwargs": model_init_kwargs,
        "feature_fields": list(feature_fields),
        "window_len": int(window_len),
        "scaler": scaler,
        "scaler_clip": float(scaler_clip),
        "threshold": 0.5,                # may be tuned later; keep default
        "metrics_val": {"loss": best_val_loss},
        # embed the resolved run_config dict if you set it in main()
        "run_config": globals().get("_LAST_RESOLVED_RUNCONFIG_DICT"),
        "version": globals().get("_LAST_RESOLVED_RUNCONFIG_DICT", {}).get("version", "v1"),
    }
    return payload



def predict_from_npz(trained: dict, paths: list[str]) -> dict:
    ff = trained["feature_fields"]
    wl = trained["window_len"]
    scaler = trained["scaler"]
    clip = float(trained.get("scaler_clip", 8.0))
    rc = trained.get("run_config", {})
    use_valid_idx = rc.get("data", {}).get("use_valid_idx", True)
    thr = float(trained.get("threshold", 0.5))

    # Rebuild model
    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_state_dict(trained["model_state"])

    # Windows (already filtered: valid-only OR all sliding)
    X, y, _, _ = make_windows_from_npz(paths, ff, window_len=wl, use_valid_idx=use_valid_idx)
    y = np.asarray(y).reshape(-1)
    X = apply_scaler(X, scaler, clip=clip)

    # Inference
    probs = []
    with torch.no_grad():
        for xb, _ in batched_loader(X, y, batch_size=512, shuffle=False):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    prob = np.concatenate(probs, axis=0).reshape(-1)

    # Align
    n = min(len(prob), len(y))
    prob = prob[:n]
    y = y[:n]
    pred = (prob >= thr).astype(int)

    # Best-effort index:
    # - If we slid across all positions, we can map to timeline as wl-1 ... wl-1+(n-1)
    # - If we used only valid positions (sparse), we don't know the original positions (not returned), so return None.
    if not use_valid_idx:
        index = np.arange(wl - 1, wl - 1 + n, dtype=np.int64)
    else:
        index = None  # unknown without valid_idx from windowing

    return {
        "index": index,          # None if unknown
        "prob": prob,            # probabilities
        "pred": pred,            # binarized with stored threshold
        "threshold": thr,
        "feature_fields": ff,
        "window_len": wl,
    }
def predict_from_npz_writeback(
    trained: dict,
    paths: list[str],
    in_place: bool = False,
    suffix: str = ".pred",
    overwrite_labels: bool = False,
    batch_size: int = 512,
    use_valid_idx_only = False
) -> list[str]:
    """
    Predict on NPZ files using a config-driven checkpoint and write results back.

    Writes per-timestep arrays:
      - 'labels_pred' (int 0/1)
      - 'pred_prob' (float)
      - 'pred_threshold' (scalar; repeated in the archive for convenience)

    If overwrite_labels=True and the NPZ contains 'labels', it will overwrite 'labels'
    with the predicted labels (use with care).

    If 'use_valid_idx' in the embedded run_config is True, we try to read 'valid_idx'
    from each NPZ to place predictions at their original positions. If 'valid_idx' is
    missing, we fall back to sliding indices (window_len-1..T-1) and print a warning.

    Returns:
        List of written output file paths (str).
    """
    import numpy as np
    import torch
    from pathlib import Path

    # --- Restore model & preprocessing from checkpoint ---
    ff        = trained["feature_fields"]
    wl        = int(trained["window_len"])
    scaler    = trained.get("scaler", None)
    clip      = float(trained.get("scaler_clip", 8.0))
    rc        = trained.get("run_config", {}) or {}
    use_valid = use_valid_idx_only #bool(rc.get("data", {}).get("use_valid_idx", True))
    thr       = float(trained.get("threshold", 0.5))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    model.load_state_dict(trained["model_state"])
    model.to(device).eval()

    out_paths = []

    for p in paths:
        p = Path(p)
        with np.load(p, allow_pickle=True) as npz:
            # --- Heuristics to discover series length ---
            # Prefer an existing 'labels' for length, else any 1D array, else last usable index
            series_len = None
            if "labels" in npz:
                series_len = int(npz["labels"].shape[0])
            else:
                # pick the longest 1D or first-dim length of a 2D array as fallback
                for k in npz.files:
                    arr = npz[k]
                    if arr.ndim == 1:
                        series_len = max(series_len or 0, int(arr.shape[0]))
                    elif arr.ndim >= 2:
                        series_len = max(series_len or 0, int(arr.shape[0]))
            if series_len is None:
                # we will derive from index later if needed
                series_len = 0

            # --- Build windows for this single file (using same pipeline as training) ---
            # We only need X (and y length alignment); call your existing helper:
            X, y, _, _ = make_windows_from_npz([str(p)], ff, window_len=wl, use_valid_idx=use_valid)
            y = np.asarray(y).reshape(-1)
            # scale features
            if scaler is not None:
                X = apply_scaler(X, scaler, clip=clip)

            # --- Inference (probabilities per window) ---
            probs = []
            with torch.no_grad():
                for xb, _ in batched_loader(X, y, batch_size=batch_size, shuffle=False):
                    xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
                    logits = model(xb)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
            prob = np.concatenate(probs, axis=0).reshape(-1)

            # --- Index mapping back to the timeline ---
            # If valid_idx exists in NPZ and we're in valid-only mode, use it.
            if use_valid and "valid_idx" in npz:
                valid_idx = np.asarray(npz["valid_idx"]).reshape(-1)
                # clean floats/NaNs if any
                if np.issubdtype(valid_idx.dtype, np.floating):
                    valid_idx = valid_idx[~np.isnan(valid_idx)]
                idx = valid_idx.astype(np.int64, copy=False)
            else:
                # slide-all mapping or fallback
                idx = np.arange(wl - 1, (wl - 1) + len(prob), dtype=np.int64)
                if use_valid and "valid_idx" not in npz:
                    print(f"[WARN] {p.name}: 'use_valid_idx' is True but 'valid_idx' not found; "
                          f"falling back to sliding indices.", flush=True)

            # Ensure non-negative and increasing
            idx = idx[(idx >= 0)]
            # If we couldn't establish series length earlier, use idx max+1 as length
            if series_len == 0 and idx.size:
                series_len = int(idx.max()) + 1

            # --- Allocate full-length arrays and place predictions ---
            prob_full = np.full(series_len, np.nan, dtype=np.float32)
            pred_full = np.full(series_len, -1,   dtype=np.int8)

            n = min(len(prob), len(idx))
            if n == 0:
                print(f"[WARN] {p.name}: no valid prediction positions; skipping write.", flush=True)
                out_paths.append(str(p))
                continue

            prob_full[idx[:n]] = prob[:n]
            pred_full[idx[:n]] = (prob[:n] >= thr).astype(np.int8)

            # --- Prepare output path ---
            if in_place:
                out_path = p
                tmp_path = p.with_suffix(p.suffix)
                target_for_save = tmp_path
            else:
                out_path = p.with_name(p.stem + suffix + p.suffix)
                target_for_save = out_path

            # --- Materialize original arrays (force copies) ---
            # Important: read arrays out of the NpzFile now, so we don't save lazy views.
            save_dict = {k: np.array(npz[k]) for k in npz.files}

            # Add threshold as scalar metadata
            save_dict["pred_threshold"] = np.array(thr, dtype=np.float32)

            # Where to put labels:
            LBL_KEY = "labels" if overwrite_labels and ("labels" in save_dict) else "labels_pred"
            save_dict[LBL_KEY] = pred_full.astype(np.int8)

            # Probabilities
            save_dict["pred_prob"] = prob_full.astype(np.float32)

            # --- Save NPZ ---
            np.savez_compressed(target_for_save, **save_dict)

            # If in-place, replace the original atomically
            if in_place:
                try:
                    tmp_path.replace(out_path)
                except Exception:
                    out_path.unlink(missing_ok=True)
                    tmp_path.replace(out_path)

            # --- Verify keys actually exist in the written file ---
            try:
                with np.load(out_path, allow_pickle=False) as chk:
                    missing = [k for k in (LBL_KEY, "pred_prob", "pred_threshold") if k not in chk.files]
                    if missing:
                        print(f"[WARN] {out_path.name}: missing keys after save: {missing}", flush=True)
                    else:
                        print(f"[OK] {out_path.name}: wrote '{LBL_KEY}' shape={chk[LBL_KEY].shape}, "
                              f"'pred_prob' shape={chk['pred_prob'].shape}, thr={float(chk['pred_threshold'])}",
                              flush=True)
            except Exception as e:
                print(f"[WARN] {out_path.name}: could not verify saved file ({e})", flush=True)

            out_paths.append(str(out_path))

    return out_paths


def evaluate_from_npz(trained: dict, eval_paths: list[str], feature_fields: list[str] | None = None) -> dict:
    # ---- Restore preprocessing & model ----
    ff = trained.get("feature_fields", feature_fields or [])
    wl = trained["window_len"]
    scaler = trained["scaler"]
    clip = float(trained.get("scaler_clip", 8.0))
    rc = trained.get("run_config", {})
    thresholds = rc.get("eval", {}).get("thresholds", [0.5])

    # Rebuild model from checkpoint
    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_state_dict(trained["model_state"])

    # ---- Build windows (already filtered per config) ----
    # IMPORTANT: make_windows_from_npz returns exactly the windows to evaluate (no extra indexing)
    X, y, _, _ = make_windows_from_npz(eval_paths, ff, window_len=wl, use_valid_idx=True)
    # If your signature only returns (X, y, T, feat_list), keep as-is; the 4th return is ignored.
    y = np.asarray(y).reshape(-1)
    X = apply_scaler(X, scaler, clip=clip)

    # Sanity
    if X.shape[0] != len(y):
        raise ValueError(f"Length mismatch after windowing: X={X.shape[0]} vs y={len(y)}")

    # ---- Inference ----
    probs = []
    with torch.no_grad():
        for xb, _ in batched_loader(X, y, batch_size=512, shuffle=False):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs, axis=0).reshape(-1)

    # Align (defensive)
    n = min(len(p), len(y))
    p = p[:n]
    y = y[:n]

    # Filter out ignored labels (-1), if used
    mask = (y >= 0)
    y_eval = y[mask]
    p_eval = p[mask]

    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
    out = {}
    try:
        out["auprc"] = float(average_precision_score(y_eval, p_eval))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["roc_auc"] = float(roc_auc_score(y_eval, p_eval))
    except Exception:
        out["roc_auc"] = float("nan")

    best_f1, best_thr = -1.0, None
    for thr in thresholds:
        yhat = (p_eval >= thr).astype(int)
        f1   = f1_score(y_eval, yhat, zero_division=0)
        acc  = accuracy_score(y_eval, yhat)
        out[f"f1@{thr:.2f}"] = float(f1)
        out[f"acc@{thr:.2f}"] = float(acc)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    out["chosen_threshold"] = trained.get("threshold", best_thr if best_thr is not None else 0.5)
    return out

def _best_threshold_on_val(trained_payload: dict, val_paths: list[str], feature_fields: list[str], grid=None):
    """
    Sweep thresholds on validation to pick the best F1 (tie-break by AUPRC).
    Returns (best_thr, metrics_dict_for_best_thr).
    """
    if grid is None:
        grid = [round(x, 2) for x in np.linspace(0.3, 0.8, 11)]
    # Reuse your evaluate_from_npz for AUPRC baseline
    base = evaluate_from_npz(trained_payload, val_paths, feature_fields)
    best = (-1.0, -1.0, 0.5)  # (f1, auprc, thr)
    from sklearn.metrics import f1_score, accuracy_score
    # We need probabilities to compute F1 per threshold → run a one-off forward pass
    ff = trained_payload["feature_fields"]
    wl = trained_payload["window_len"]
    scaler = trained_payload["scaler"]
    clip = float(trained_payload.get("scaler_clip", 8.0))

    model = TinyTemporalCNN(**trained_payload["model_init_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_state_dict(trained_payload["model_state"])

    X, y, _, _ = make_windows_from_npz(val_paths, ff, window_len=wl, use_valid_idx=True)
    y = np.asarray(y).reshape(-1)
    X = apply_scaler(X, scaler, clip=clip)
    probs = []
    with torch.no_grad():
        for xb, _ in batched_loader(X, y, batch_size=512, shuffle=False):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs, axis=0).reshape(-1)

    # Align and mask ignored labels
    n = min(len(p), len(y))
    y = y[:n]; p = p[:n]
    mask = (y >= 0)
    y_eval, p_eval = y[mask], p[mask]

    for thr in grid:
        yhat = (p_eval >= thr).astype(int)
        f1 = f1_score(y_eval, yhat, zero_division=0)
        # prefer the threshold that maximizes F1; break ties by AUPRC (from base)
        cand = (f1, base.get("auprc", float("nan")), thr)
        if cand > best:
            best = cand

    best_thr = best[2]
    out = dict(best_threshold=best_thr, auprc=base.get("auprc", float("nan")))
    return best_thr, out


# ----------------------------
# Serialization & Inference wrapper
# ----------------------------
class BounceDetector:
    """Inference wrapper for use from any script.
    - predict_window(window_np): window_np shape [T,S=1,F] or [B,T,S=1,F]
    Returns probabilities in [0,1].
    """
    def __init__(self, model: nn.Module, feature_fields: List[str], window_len: int, scaler, device: str | None = None):
        self.model = model
        self.scaler = scaler
        self.feature_fields = feature_fields
        self.window_len = window_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        if X.ndim == 3:
            X = X[None, ...]
        assert X.ndim == 4, f"Expected [B,T,S,F], got {X.shape}"
        return torch.from_numpy(X).float().to(self.device)

    @torch.no_grad()
    def predict_window(self, window_np: np.ndarray) -> np.ndarray:
        # Optional fill/scale for safety
        if self.scaler is not None:
            if window_np.ndim == 3:
                # Ensure shape [T, S, F]
                if window_np.shape[1] != 1 and window_np.shape[2] == 1:
                    # It’s likely [T, F, 1], so swap
                    window_np = np.transpose(window_np, (0, 2, 1))
                window_np = ffill_bfill_time_batch(window_np[None, ...])[0]
            elif window_np.ndim == 4:
                if window_np.shape[2] != 1 and window_np.shape[3] == 1:
                    window_np = np.transpose(window_np, (0, 1, 3, 2))
                window_np = ffill_bfill_time_batch(window_np)
            window_np = apply_scaler(window_np, self.scaler, clip=8.0)
        else:
            window_np = np.nan_to_num(window_np, nan=0.0, posinf=8.0, neginf=-8.0)

        x = self._to_tensor(window_np)
        logits = self.model(x).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def save_trained(trained: dict, path: str | Path) -> Path:
    """
    Save a self-contained checkpoint. Also writes sidecar JSONs for inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [k for k in _REQUIRED_KEYS if k not in trained]
    if missing:
        raise ValueError(f"save_trained: payload missing keys: {missing}")

    # Ensure run_config present (at least minimal)
    trained.setdefault("run_config", trained.get("run_config", {
        "version": trained.get("version", "v1"),
        "data": {"feature_fields": trained.get("feature_fields", []),
                 "window_len": trained.get("window_len", None)}
    }))

    # Save checkpoint
    with open(path, "wb") as f:
        pickle.dump(trained, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional: sidecars for readability (don’t affect loading)
    try:
        side = path.with_suffix(path.suffix + ".meta")
        side.mkdir(exist_ok=True)
        (side / "run_config.json").write_text(
            json.dumps(trained.get("run_config", {}), indent=2, ensure_ascii=False)
        )
        (side / "model_init_kwargs.json").write_text(
            json.dumps(trained["model_init_kwargs"], indent=2, ensure_ascii=False)
        )
    except Exception:
        # Sidecars are best-effort—ignore failures
        pass

    return path



def _choose_device(device: str):
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _ensure_model_kwargs(payload: dict) -> dict:
    """
    Return a complete model_init_kwargs without touching existing values.
    Only fills missing fields using legacy payload hints or sane defaults.
    """
    mk = dict(payload.get("model_init_kwargs") or {})

    # Required
    if "F" not in mk:
        ff = payload.get("feature_fields", [])
        mk["F"] = int(len(ff)) if ff else int(payload.get("F", 0))
    if "T" not in mk:
        mk["T"] = int(payload.get("window_len"))

    # Safe defaults for optional fields
    mk.setdefault("hidden", 64)
    mk.setdefault("n_out", 1)
    mk.setdefault("k", 9)
    mk.setdefault("layers", 2)
    mk.setdefault("dropout", 0.1)

    # Final sanity
    if not mk.get("F") or not mk.get("T"):
        raise ValueError("model_init_kwargs requires F and T; could not infer from payload.")
    return mk

def load_trained(path: str | Path, build_model: bool = False, device: str = "auto"):
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    # Normalize/repair only if incomplete
    payload = dict(payload)
    payload.setdefault("feature_fields", payload.get("feature_fields", []))
    payload.setdefault("run_config", payload.get("run_config", {}))
    payload.setdefault("version", payload.get("version", payload["run_config"].get("version", "v1")))
    payload.setdefault("threshold", payload.get("threshold", 0.5))
    payload.setdefault("scaler_clip", payload.get("scaler_clip", 8.0))
    if "window_len" not in payload and "T" in payload:
        payload["window_len"] = int(payload["T"])

    # Ensure model kwargs *once* (does not overwrite existing keys)
    payload["model_init_kwargs"] = _ensure_model_kwargs(payload)

    if not build_model:
        return payload

    # Build model from the now-complete kwargs
    mk = payload["model_init_kwargs"]
    model = TinyTemporalCNN(**mk)
    dev = torch.device("cuda" if (device in (None, "auto") and torch.cuda.is_available()) else (device or "cpu"))
    model.to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return payload, model, dev


def create_bouncedetector(path: str, device: str | None = None) -> BounceDetector:
    """
    Create a BounceDetector from a saved checkpoint using the config-driven loader.
    - Uses load_trained(...) to restore payload + model (on chosen device).
    - Wires scaler/metadata into the detector for consistent preprocessing.
    """
    # load_trained should be the improved version that can build the model:
    #   payload, model, dev = load_trained(path, build_model=True, device=device_or_auto)
    payload, model, dev = load_trained(path, build_model=True, device=("auto" if device is None else device))

    # Sanity: feature count should match model init kwargs
    ff = payload.get("feature_fields", [])
    mk = payload.get("model_init_kwargs", {})
    if mk and "F" in mk and len(ff) and mk["F"] != len(ff):
        print(f"[WARN] feature_fields length ({len(ff)}) != model F ({mk['F']}). "
              f"Check the checkpoint or feature configuration.", flush=True)

    det = BounceDetector(
        model=model,
        feature_fields=ff,
        window_len=payload.get("window_len"),
        device=str(dev),
        scaler=payload.get("scaler", None),
    )

    # Attach useful metadata (harmless if BounceDetector doesn't use them internally)
    setattr(det, "threshold", float(payload.get("threshold", 0.5)))
    setattr(det, "scaler_clip", float(payload.get("scaler_clip", 8.0)))
    setattr(det, "run_config", payload.get("run_config", {}))
    setattr(det, "model_init_kwargs", mk)

    return det

###############################             Serch               ###########################################
def _sample_search_space(trial, rc: RunConfig):
    """Define a compact, sane search space."""
    space = {}
    # Data
    space["window_len"] = trial.suggest_categorical("data.window_len", [32, 64, 128, 256, 512, 1024])
    # Model
    space["hidden"]  = trial.suggest_categorical("model.hidden", [8, 16, 32, 64, 128, 256])
    space["k"]       = trial.suggest_categorical("model.k", [3, 7, 9, 13, 15])
    space["layers"]  = trial.suggest_categorical("model.layers", [1, 2, 3, 4 ,5])
    space["dropout"] = trial.suggest_float("model.dropout", 0.0, 0.5, step=0.05)
    # Train
    space["lr"]           = trial.suggest_float("train.lr", 1e-4, 1e-3, log=True)
    space["weight_decay"] = trial.suggest_float("train.weight_decay", 1e-6, 1e-3, log=True)
    space["batch_size"]   = trial.suggest_categorical("train.batch_size", [128, 256, 512, 1024])
    # Class-weight around auto
    cw_mode = trial.suggest_categorical("train.class_weight_mode", ["auto", "half", "double"])
    space["cw_mode"] = cw_mode
    return space

def _resolve_trial_configs(base: RunConfig, space: dict) -> RunConfig:
    """Create a shallow copy of base RunConfig with sampled overrides."""
    # copy dataclasses
    data = deepcopy(_asdict_dc(base.data));  model = deepcopy(_asdict_dc(base.model))
    train = deepcopy(_asdict_dc(base.train)); evalc = deepcopy(_asdict_dc(base.eval) if base.eval else {})
    # apply overrides
    data["window_len"] = space["window_len"]
    model.update(dict(hidden=space["hidden"], k=space["k"], layers=space["layers"], dropout=space["dropout"], window_len=space["window_len"]))
    train.update(dict(lr=space["lr"], weight_decay=space["weight_decay"], batch_size=space["batch_size"]))
    # pack back
    return RunConfig(data=DataConfig(**data), model=ModelConfig(**model), train=TrainConfig(**train), eval=EvalConfig(**evalc) if evalc else None, version=base.version, git_commit=base.git_commit)

def _compute_cw_factor(cw_mode: str, auto_ratio: float) -> float | str:
    if cw_mode == "auto":   return "auto"
    if cw_mode == "half":   return max(0.5 * auto_ratio, 1.0)
    if cw_mode == "double": return 2.0 * auto_ratio
    return "auto"

def run_search_optuna(
    base_rc: RunConfig,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    out_dir: str,
    n_trials: int = 50,
    study_name: str = "bounce_search",
    storage: str | None = None,  # e.g. "sqlite:///bounce_search.db"
):
    try:
        import optuna
        from optuna.pruners import SuccessiveHalvingPruner
        from optuna.samplers import TPESampler
    except Exception as e:
        raise SystemExit("Optuna is required: pip install optuna") from e

    sampler = TPESampler(seed=base_rc.data.seed)
    pruner = SuccessiveHalvingPruner()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Precompute auto class-weight ratio once from train y
    Xtmp, ytmp, _, _ = make_windows_from_npz(train_files, base_rc.data.feature_fields, base_rc.data.window_len, use_valid_idx=True)
    yflat = ytmp[ytmp >= 0]
    pos = max(1, int((yflat == 1).sum())); neg = max(1, int((yflat == 0).sum()))
    auto_ratio = float(neg / pos)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: "optuna.trial.Trial") -> float:
        space = _sample_search_space(trial, base_rc)
        rc = _resolve_trial_configs(base_rc, space)

        # class weight for this trial
        cw = _compute_cw_factor(space["cw_mode"], auto_ratio)

        # Build model per trial
        model = TinyTemporalCNN(
            F=len(rc.data.feature_fields or []),
            T=rc.data.window_len,
            hidden=rc.model.hidden,
            n_out=rc.model.out_dim,
            k=rc.model.k,
            layers=rc.model.layers,
            dropout=rc.model.dropout,
        )

        globals()['_LAST_RESOLVED_RUNCONFIG_DICT'] = _asdict_dc(rc)  # embed in payload
        device = _device_from_cfg(rc.train)
        trained = train_from_npz(
            train_paths=train_files,
            val_paths=val_files,
            feature_fields=rc.data.feature_fields,
            window_len=rc.data.window_len,
            epochs=rc.train.epochs,
            batch_size=rc.train.batch_size,
            lr=rc.train.lr,
            weight_decay=rc.train.weight_decay,
            device=device,
            class_weight=cw,
            precision16=rc.train.precision16,
            model=model,
            scaler_clip=rc.data.scaler_clip,
            early_stopping_patience=rc.train.early_stopping_patience,
        )

        # Pick threshold on validation and record best AUPRC
        best_thr, thr_info = _best_threshold_on_val(trained, val_files, rc.data.feature_fields)
        trained["threshold"] = best_thr

        # Save trial artifacts
        tdir = out_dir / f"trial_{trial.number:04d}"
        tdir.mkdir(parents=True, exist_ok=True)
        save_trained(trained, tdir / "model.ckpt")
        (tdir / "search_space.json").write_text(json.dumps(space, indent=2))
        (tdir / "run_config.json").write_text(json.dumps(_asdict_dc(rc), indent=2))

        # Report to Optuna (so pruner can act if you later add intermediate reports)
        score = float(thr_info.get("auprc", float("nan")))
        trial.set_user_attr("best_threshold", best_thr)
        trial.set_user_attr("val_auprc", score)
        return score

    study.optimize(objective, n_trials=n_trials)

    # Write a tiny summary
    (Path(out_dir) / "study_summary.json").write_text(
        json.dumps({
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_attrs": study.best_trial.user_attrs,
            "n_trials": len(study.trials),
        }, indent=2)
    )
    return study


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse, pickle
    parser = argparse.ArgumentParser(description="Train, save, and evaluate bounce detector from NPZ files (config-driven).")
    parser.add_argument("--config", type=str, required=False, help="Path to JSON config with data/model/train/eval blocks.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config keys: key.sub=value (repeatable)")
    parser.add_argument("--out", required=True, help="Path to save model (e.g. models/bounce.ckpt)")
    parser.add_argument("--device", default=None, help="cuda|cpu|auto (overrides train.device)")
    parser.add_argument("--score-test", action="store_true", help="Score Test")
    # --- Prediction / write-back flags (clean descriptions) ---
    parser.add_argument("--predict-back", action="store_true",
        help="One or more NPZ paths or globs to run inference on and write predictions back to the files and add pred.npz as extension."    )
    parser.add_argument("--predict-in-place", action="store_true",
                        help="Overwrite the input NPZ file(s). If not set, outputs are written to the original file but into labels_pred and labels_prob structure.")
    parser.add_argument("--predict-overwrite-labels", action="store_true", help="If set, write predicted labels into the existing 'labels' array. "
             "If not set (default), write predictions to 'labels_pred' and keep original 'labels' unchanged.")
    parser.add_argument("--predict-batch", type=int, default=512, help="Batch size to use during prediction.")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to a trained checkpoint (.ckpt/.pkl). Required for --predict-back unless --out points to a checkpoint file.")
    parser.add_argument("--only-valid_idx", type=bool, default=False, help="predict only on valid_idxes.")

    parser.add_argument("--search", type=int, default=0, help="Number of Optuna trials (0 = no search)")
    parser.add_argument("--study-name", type=str, default="bounce_search")
    parser.add_argument("--storage", type=str, default=None, help='Optuna storage, e.g. sqlite:///bounce.db')

    args = parser.parse_args()

    # Load or synthesize config
    cfg_dict = {}
    if args.config:
        cfg_dict = _load_config_json(args.config)
    else:
        raise SystemExit("--config missing")

    # Apply --set overrides
    for ov in args.overrides or []:
        if "=" not in ov:
            print(f"[WARN] Ignoring malformed override: {ov}")
            continue
        k, v = ov.split("=", 1)
        _merge_dotset(cfg_dict, k.strip(), v.strip())

    # Allow --device override
    if args.device:
        cfg_dict.setdefault("train", {})["device"] = args.device

    # Resolve dataclasses
    rc = _resolve_run_config(cfg_dict)

    # Expand globs
    train_files = _expand_globs(rc.data.train_glob)
    val_files   = _expand_globs(rc.data.val_glob)
    test_files  = _expand_globs(rc.data.test_glob)
    if not train_files or not val_files or not test_files:
        raise SystemExit("Train/Val/Test files must be specified in config or flags.")

    # Features
    feature_fields = rc.data.feature_fields or []
    if not feature_fields:
        raise SystemExit("feature_fields must be provided in config.data.feature_fields.")

    # Persist resolved config in checkpoint
    globals()['_LAST_RESOLVED_RUNCONFIG_DICT'] = _asdict_dc(rc)

    # Build and train
    device = rc.train.device if rc.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(
        F=len(feature_fields), T=rc.data.window_len,
        hidden=rc.model.hidden, n_out=rc.model.out_dim,
        k=rc.model.k, layers=rc.model.layers, dropout=rc.model.dropout
    )

    if args.score_test and args.checkpoint:
        # payload, model, dev = load_trained(args.checkpoint, build_model=True, device="auto")
        # res = score_silent_lowhunter(
        #     trained=payload,
        #     full_paths=val_files[:3],  # run on your val files as full sessions
        #     price_field="mid",
        #     low_radius=20,
        #     max_pos_per_session=None,
        #     thr_grid=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        #     weights={"hit": 1.0, "outside": 4.0, "chatter": 2.0, "auprc": 0.2}
        # )
        # print("Threshold  Score   HitRate  OutsideRate  PosRate")
        # for row in sorted(res["by_threshold"], key=lambda r: r["score"], reverse=True)[:5]:
        #     print(f"{row['thr']:8.2f}  {row['score']:6.3f}  {row['hit_rate']:7.3f}  "
        #           f"{row['outside_rate']:11.5f}  {row['pos_rate']:7.5f}")
        # print("Best:", res["best_threshold"], "Score:", res["score"])
        #
        # payload["threshold"] = res["best_threshold"]
        # print("Custom score:", res["score"], "best thr:", res["best_threshold"])
        raise SystemExit(0)
    if args.score_test and not args.checkpoint:
        print("Need a Checkpoint --checkpoint ...")
        raise SystemExit(0)

    # --- Prediction-only path with write-back ---
    if args.predict_back or args.predict_in_place:
        # Resolve checkpoint
        ckpt = args.checkpoint
        if ckpt is None:
            # If --out points to a file (ending .ckpt), use it
            from pathlib import Path

            p_out = Path(args.out)
            if p_out.is_file() and p_out.suffix in {".ckpt", ".pkl", ".pickle", ".npz"}:
                ckpt = str(p_out)
            else:
                raise SystemExit(
                    "For --predict-back, provide --checkpoint path/to.ckpt (or set --out to a checkpoint file).")

        # Load checkpoint and build model on auto device
        payload, model, dev = load_trained(ckpt, build_model=True, device="auto")
        # Reuse the trained payload; model/dev are already inside it and not needed by the helper
        # (predict helper reconstructs its own model from payload to keep things isolated).
        written = predict_from_npz_writeback(
            trained=payload,
            paths=test_files,
            in_place=args.predict_in_place,
            suffix=".pred",
            overwrite_labels=args.predict_overwrite_labels,
            batch_size=args.predict_batch,
            use_valid_idx_only=args.only_valid_idx
        )
        print("Wrote:", *written, sep="\n  ")
        raise SystemExit(0)

    if args.search and args.search > 0:
        # Resolve dataclasses already done → rc, train_files, val_files, test_files, feature_fields
        print(f"Starting parameter search for {args.search} trials...")
        study = run_search_optuna(
            base_rc=rc,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            out_dir=args.out,  # a directory to store trial_XXXX
            n_trials=args.search,
            study_name=args.study_name,
            storage=args.storage,
        )
        print("Best value (val AUPRC):", study.best_value)
        print("Best params:", study.best_trial.params)
        raise SystemExit(0)


    trained = train_from_npz(
        train_paths=train_files,
        val_paths=val_files,
        feature_fields=feature_fields,
        window_len=rc.data.window_len,
        epochs=rc.train.epochs,
        batch_size=rc.train.batch_size,
        lr=rc.train.lr,
        weight_decay=rc.train.weight_decay,
        device=device,
        class_weight=rc.train.class_weight_pos,
        precision16=rc.train.precision16,
        model=model
    )

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_trained(trained, out_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate
    trained = load_trained("models/bounce.ckpt")  # returns payload dict
    metrics = evaluate_from_npz(trained, test_files, feature_fields)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("Done.")

