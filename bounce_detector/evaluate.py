from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
from .data import make_windows_from_npz, apply_scaler
from .model import TinyTemporalCNN
from .train import batched_loader

def evaluate_from_npz(trained: dict, eval_paths: list[str], feature_fields: list[str] | None = None) -> dict:
    ff = trained.get("feature_fields", feature_fields or [])
    wl = trained["window_len"]
    scaler = trained["scaler"]
    clip = float(trained.get("scaler_clip", 8.0))

    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_state_dict(trained["model_state"])

    X, y, _, _ = make_windows_from_npz(eval_paths, ff, window_len=wl, use_valid_idx=True)
    y = np.asarray(y).reshape(-1)
    X = apply_scaler(X, scaler, clip=clip)

    if X.shape[0] != len(y):
        raise ValueError(f"Length mismatch after windowing: X={X.shape[0]} vs y={len(y)}")

    probs = []
    with torch.no_grad():
        for xb, _ in batched_loader(X, y, batch_size=512, shuffle=False):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs, axis=0).reshape(-1)

    n = min(len(p), len(y))
    p = p[:n]
    y = y[:n]
    mask = (y >= 0)
    y_eval = y[mask]
    p_eval = p[mask]

    out = {}
    try:
        out["auprc"] = float(average_precision_score(y_eval, p_eval))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["roc_auc"] = float(roc_auc_score(y_eval, p_eval))
    except Exception:
        out["roc_auc"] = float("nan")

    for thr in [0.5]:
        yhat = (p_eval >= thr).astype(int)
        out[f"f1@{thr:.2f}"] = float(f1_score(y_eval, yhat, zero_division=0))
        out[f"acc@{thr:.2f}"] = float(accuracy_score(y_eval, yhat))

    out["chosen_threshold"] = trained.get("threshold", 0.5)
    return out

def best_threshold_on_val(trained_payload: dict, val_paths: list[str], feature_fields: list[str], grid=None):
    if grid is None:
        grid = [round(x, 2) for x in np.linspace(0.3, 0.8, 11)]
    from .data import apply_scaler, make_windows_from_npz
    from .model import TinyTemporalCNN

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

    n = min(len(p), len(y))
    y = y[:n]; p = p[:n]
    mask = (y >= 0)
    y_eval, p_eval = y[mask], p[mask]

    best_f1, best_thr = -1.0, None
    for thr in grid:
        yhat = (p_eval >= thr).astype(int)
        f1 = f1_score(y_eval, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, {"auprc": float(average_precision_score(y_eval, p_eval)) if y_eval.size else float("nan")}
