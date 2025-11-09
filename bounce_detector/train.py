from __future__ import annotations

import random
from typing import Iterable, Tuple
import numpy as np
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as Ft
from torch.amp import autocast, GradScaler
from .data import make_windows_from_npz, fit_scaler_nan_safe, apply_scaler
from .config import RunConfig

def batched_loader(X, y=None, batch_size: int = 256, shuffle: bool = False, drop_last: bool = False):
    try:
        N = len(X)
    except TypeError:
        raise ValueError("X must be indexable with a defined length.")
    if y is not None and len(y) != N:
        raise ValueError(f"Length mismatch: len(X)={N} but len(y)={len(y)}")
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    end = (N // batch_size) * batch_size if drop_last else N
    for start in range(0, end, batch_size):
        stop = min(start + batch_size, N)
        sel = idx[start:stop]
        xb = X[sel]
        yb = None if y is None else y[sel]
        yield xb, yb

def _device_from_cfg(dev: str) -> str:
    if dev == "auto" or dev is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev



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
    model: nn.Module,
    scaler_clip: float = 8.0,
    early_stopping_patience: int = 10,
    grad_clip_max_norm: float | None = None,
    warmup_steps: int = 0,
    use_cosine_after_warmup: bool = False,
) -> dict:
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True  # OK since shapes are constant per window_len
    device = torch.device(_device_from_cfg(device))
    model.to(device)

    # --- data ---
    Xtrain, ytrain, _, _ = make_windows_from_npz(
        train_paths, feature_fields, window_len=window_len)
    Xval, yval, _, _ = make_windows_from_npz(
        val_paths, feature_fields, window_len=window_len)
    print("loaded windows")

    # --- scale ---
    scaler_feat = fit_scaler_nan_safe(Xtrain)
    Xtrain = apply_scaler(Xtrain, scaler_feat, clip=scaler_clip)
    Xval   = apply_scaler(Xval,   scaler_feat, clip=scaler_clip)

    # --- class weight / pos_weight ---
    cw = class_weight
    if isinstance(cw, str) and cw == "auto":
        y_flat = ytrain[ytrain >= 0]                 # ignore invalid labels for ratio
        pos = max(1, int((y_flat == 1).sum()))
        neg = max(1, int((y_flat == 0).sum()))
        cw = float(neg / pos)
    pos_weight = torch.tensor([float(cw)], device=device, dtype=torch.float32)

    # --- loss (masked for invalid labels, e.g., -1) ---
    def bce_logits_loss_masked(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets expected shape (N,1) with {0,1} or -1 for invalid
        mask = (targets >= 0.0)
        if not mask.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        loss = Ft.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        loss = loss[mask]
        return loss.mean()

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional warmup + cosine schedule
    if warmup_steps > 0 or use_cosine_after_warmup:
        total_steps = epochs * math.ceil(len(Xtrain) / max(1, batch_size))
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            if use_cosine_after_warmup:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler_amp = GradScaler(enabled=(precision16 and device.type == "cuda"))
    best_val_loss = float("inf")
    best_state = None
    patience = early_stopping_patience

    # --- training loop ---
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        for xb, yb in batched_loader(Xtrain, ytrain, batch_size):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)

            valid = (yb >= 0.0)
            if not valid.any():
                continue  # << skip this batch
            optim.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(precision16 and device.type == "cuda")):
                logits = model(xb)
                loss = bce_logits_loss_masked(logits, yb)

            scaler_amp.scale(loss).backward()

            if grad_clip_max_norm is not None:
                scaler_amp.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            scaler_amp.step(optim)
            scaler_amp.update()

            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item()) * xb.size(0)
            n_train += xb.size(0)
            global_step += 1

        train_loss = running_loss / max(1, n_train)

        # --- validation ---
        model.eval()
        with torch.no_grad():
            vloss_sum, n_val = 0.0, 0
            with autocast(device_type="cuda", enabled=(precision16 and device.type == "cuda")):
                for xb, yb in batched_loader(Xval, yval, batch_size):
                    xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
                    yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)
                    logits = model(xb)
                    vloss = bce_logits_loss_masked(logits, yb)
                    vloss_sum += float(vloss.item()) * xb.size(0)
                    n_val += xb.size(0)
        vloss_epoch = vloss_sum / max(1, n_val)

        print(f"epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_loss: {vloss_epoch:.6f}")

        # --- early stopping ---
        if vloss_epoch < best_val_loss:
            best_val_loss = vloss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch} (no val improvement for {early_stopping_patience} checks).")
                break

    # restore best weights (best_state is on CPU)
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- build payload ---
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
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},  # ensure CPU tensors
        "model_init_kwargs": model_init_kwargs,
        "feature_fields": list(feature_fields),
        "window_len": int(window_len),
        "scaler": scaler_feat,
        "scaler_clip": float(scaler_clip),
        "threshold": 0.5,
        "metrics_val": {"loss": float(best_val_loss)},
        "pos_weight": float(pos_weight.item()),
        "amp": bool(precision16 and device.type == "cuda"),
    }
    return payload

def stream_windows_from_files(paths, feature_fields, window_len, batch_size, scaler, clip, shuffle=False):
    # yields (xb, yb) batches already scaled; loads one file at a time
    for p in paths:
        X, y, _, _ = make_windows_from_npz([p], feature_fields, window_len=window_len)
        X = apply_scaler(X, scaler, clip=clip)
        if shuffle:
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
        for xb, yb in batched_loader(X, y, batch_size):
            yield xb, yb

def compute_scaler_and_class_weight_streaming(paths, feature_fields, window_len, sample_limit_per_file=None):
    # Online mean/std and pos/neg counts, NaN-safe and label -1 safe.
    # Returns (scaler_like_fit_scaler_nan_safe, cw_float)
    cnt = None
    mean = None
    M2 = None
    pos = 1
    neg = 1

    for p in paths:
        X, y, _, _ = make_windows_from_npz([p], feature_fields, window_len=window_len)
        if sample_limit_per_file is not None and len(X) > sample_limit_per_file:
            sel = np.random.choice(len(X), size=sample_limit_per_file, replace=False)
            Xs, ys = X[sel], y[sel]
        else:
            Xs, ys = X, y

        # update class counts (ignore invalid labels)
        y_flat = ys[ys >= 0]
        pos += int((y_flat == 1).sum())
        neg += int((y_flat == 0).sum())

        # flatten windows to rows to get per-feature stats (T*F treated as features)
        # If your scaler is featurewise across the last axis only, keep that behavior:
        # here we compute stats per last-axis feature across all time steps.
        # shape: (N, T, F)
        Xf = Xs  # (N,T,F)
        # nan-safe reduce over N and T
        # mean over (N,T)
        mu = np.nanmean(Xf, axis=(0,1))  # (F,)
        var = np.nanvar(Xf, axis=(0,1))  # (F,)
        n_valid = np.isfinite(Xf).sum(axis=(0,1))  # (F,)

        if cnt is None:
            cnt = n_valid.astype(np.float64)
            mean = mu.astype(np.float64)
            M2   = var.astype(np.float64) * cnt  # since var = M2 / n
        else:
            # parallel variance merge per feature
            nA = cnt
            nB = n_valid.astype(np.float64)
            delta = mu - mean
            tot = nA + nB
            # avoid divide-by-zero
            mask = (nB > 0)
            mean[mask] = mean[mask] + delta[mask] * (nB[mask] / np.maximum(1.0, tot[mask]))
            M2[mask] = M2[mask] + (var[mask] * nB[mask]) + (delta[mask]**2) * (nA[mask] * nB[mask] / np.maximum(1.0, tot[mask]))
            cnt = tot

    eps = 1e-8
    std = np.sqrt(np.maximum(M2 / np.maximum(1.0, cnt), 0.0)) + eps
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "eps": float(eps)}
    cw = float(neg / pos)
    return scaler, cw

def train_from_npz_streaming(
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
    model: nn.Module,
    scaler_clip: float = 8.0,
    early_stopping_patience: int = 10,
    grad_clip_max_norm: float | None = None,
    warmup_steps: int = 0,
    use_cosine_after_warmup: bool = False,
    shuffle_each_epoch: bool = True,
) -> dict:
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device(_device_from_cfg(device))
    model.to(device)

    # --- Load and scale once ---
    Xtrain, ytrain, _, _ = make_windows_from_npz(train_paths, feature_fields, window_len=window_len)
    Xval, yval, _, _ = make_windows_from_npz(val_paths, feature_fields, window_len=window_len)
    print("loaded windows")

    scaler_feat = fit_scaler_nan_safe(Xtrain)
    Xtrain = apply_scaler(Xtrain, scaler_feat, clip=scaler_clip)
    Xval   = apply_scaler(Xval,   scaler_feat, clip=scaler_clip)

    # --- class weight ---
    cw = class_weight
    if isinstance(cw, str) and cw == "auto":
        y_flat = ytrain[ytrain >= 0]
        pos = max(1, int((y_flat == 1).sum()))
        neg = max(1, int((y_flat == 0).sum()))
        cw = float(neg / pos)
    pos_weight = torch.tensor([float(cw)], device=device, dtype=torch.float32)

    def bce_logits_loss_masked(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
        mask = (targets >= 0.0)
        if not mask.any():
            return torch.zeros((), device=logits.device), 0
        loss = Ft.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        loss = loss[mask]
        return loss.mean(), int(mask.sum())

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if warmup_steps > 0 or use_cosine_after_warmup:
        total_steps = epochs * math.ceil(len(Xtrain) / max(1, batch_size))
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            if use_cosine_after_warmup:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler_amp = GradScaler(enabled=(precision16 and device.type == "cuda"))
    best_val_loss = float("inf")
    best_state = None
    patience = early_stopping_patience
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, n_train = 0.0, 0

        if shuffle_each_epoch:
            train_paths_epoch = train_paths.copy()
            random.shuffle(train_paths_epoch)
        else:
            train_paths_epoch = train_paths

        for xb, yb in batched_loader(Xtrain, ytrain, batch_size=batch_size, shuffle=True):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)
            if not (yb >= 0.0).any():
                continue

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(precision16 and device.type == "cuda")):
                logits = model(xb)
                loss, vcount = bce_logits_loss_masked(logits, yb)

            scaler_amp.scale(loss).backward()

            if grad_clip_max_norm is not None:
                scaler_amp.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            scaler_amp.step(optim)
            scaler_amp.update()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += float(loss.item()) * vcount
            n_train += vcount
            global_step += 1

        train_loss = train_loss_sum / max(1, n_train)

        # --- VAL ---
        model.eval()
        vloss_sum, n_val = 0.0, 0
        with torch.no_grad(), autocast(device_type="cuda", enabled=(precision16 and device.type == "cuda")):
            for xb, yb in batched_loader(Xval, yval, batch_size=batch_size, shuffle=False):
                xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
                yb = torch.as_tensor(yb, device=device, dtype=torch.float32).unsqueeze(1)
                logits = model(xb)
                vloss, vcount = bce_logits_loss_masked(logits, yb)
                if vcount == 0:
                    continue
                vloss_sum += float(vloss.item()) * vcount
                n_val += vcount
        vloss_epoch = vloss_sum / max(1, n_val)

        print(f"epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_loss: {vloss_epoch:.6f}")

        if vloss_epoch < best_val_loss:
            best_val_loss = vloss_epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    F = getattr(model, "F", len(scaler_feat["mean"]))
    T = window_len
    model_init_kwargs = dict(
        F=F, T=T,
        hidden=int(getattr(model, "hidden", 64)),
        n_out=1,
        k=int(getattr(model, "k", 9)),
        layers=int(getattr(model, "layers", 2)),
        dropout=float(getattr(model, "dropout", 0.1)),
    )
    payload = {
        "model_class": "TinyTemporalCNN",
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_init_kwargs": model_init_kwargs,
        "feature_fields": list(feature_fields),
        "window_len": int(window_len),
        "scaler": scaler_feat,
        "scaler_clip": float(scaler_clip),
        "threshold": 0.5,
        "metrics_val": {"loss": float(best_val_loss)},
        "pos_weight": float(pos_weight.item()),
        "amp": bool(precision16 and device.type == "cuda"),
    }
    return payload