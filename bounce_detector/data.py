from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

def load_one_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        sampled = npz["sampled"]
        labels = npz["labels"]
    return dict(sampled=sampled, labels=labels)

def make_windows_from_npz(npz_paths: List[str], feature_fields: List[str], window_len: int) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    X_list, Y_list = [], []
    for p in npz_paths:
        data = load_one_npz(p)
        sampled = data["sampled"]
        labels = data["labels"]
        feats = np.stack([sampled[name] for name in feature_fields], axis=1)
        T_total = feats.shape[0]
        for t0 in range(window_len - 1, T_total):
            X_list.append(feats[t0 - window_len + 1 : t0 + 1][None, ...])
            Y_list.append(labels[t0])
    if not X_list:
        raise RuntimeError("No training windows were constructed. Check window_len and .npz files.")
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

def _ffill_bfill_2d(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    T, C = X.shape
    mask = np.isfinite(X)
    idx = np.where(mask, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    X_ff = X.copy()
    miss = ~mask
    if miss.any():
        rows, cols = np.nonzero(miss)
        X_ff[rows, cols] = X_ff[idx[rows, cols], cols]
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
    if X_train.ndim != 4:
        raise ValueError("Expected X_train with shape [N,T,S,F]")
    N, T, S, F = X_train.shape
    Xc = X_train.reshape(N * T, S, F)
    mean = np.nanmean(Xc, axis=0)
    std = np.nanstd(Xc, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"mean": mean, "std": std}

def apply_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray], clip: float = 8.0) -> np.ndarray:
    mean, std = scaler["mean"], scaler["std"]
    if X.ndim in (3,4):
        Xn = (X - mean) / (std + 1e-9)
    else:
        raise ValueError(f"Unexpected shape for apply_scaler: {X.shape}")
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        Xn = np.clip(Xn, -clip, clip)
    return Xn