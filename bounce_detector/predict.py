from __future__ import annotations
import numpy as np
import torch
from .data import make_windows_from_npz, apply_scaler
from .model import TinyTemporalCNN
from .train import batched_loader


def predict_from_npz(trained: dict, paths: list[str]) -> dict:
    ff = trained["feature_fields"]
    wl = trained["window_len"]
    scaler = trained["scaler"]
    clip = float(trained.get("scaler_clip", 8.0))
    thr = float(trained.get("threshold", 0.5))

    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.load_state_dict(trained["model_state"])

    X, y, _, _ = make_windows_from_npz(paths, ff, window_len=wl)
    y = np.asarray(y).reshape(-1)
    X = apply_scaler(X, scaler, clip=clip)

    probs = []
    with torch.no_grad():
        for xb, _ in batched_loader(X, y, batch_size=512, shuffle=False):
            xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    prob = np.concatenate(probs, axis=0).reshape(-1)

    n = min(len(prob), len(y))
    prob = prob[:n]
    y = y[:n]
    pred = (prob >= thr).astype(int)

    index = np.arange(wl - 1, wl - 1 + n, dtype=np.int64)

    return {"index": index, "prob": prob, "pred": pred, "threshold": thr, "feature_fields": ff, "window_len": wl}

def predict_from_npz_writeback(
    trained: dict,
    paths: list[str],
    in_place: bool = False,
    suffix: str = ".pred",
    overwrite_labels: bool = False,
    batch_size: int = 512,
) -> list[str]:
    import numpy as np
    import torch
    from pathlib import Path
    from .data import make_windows_from_npz, apply_scaler
    from .model import TinyTemporalCNN
    from .train import batched_loader

    ff        = trained["feature_fields"]
    wl        = int(trained["window_len"])
    scaler    = trained.get("scaler", None)
    clip      = float(trained.get("scaler_clip", 8.0))
    thr       = float(trained.get("threshold", 0.5))
    print(f"using Threshold: {thr}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(**trained["model_init_kwargs"])
    model.load_state_dict(trained["model_state"])
    model.to(device).eval()

    out_paths = []

    for p in paths:
        p = Path(p)
        with np.load(p, allow_pickle=True) as npz:
            series_len = 0
            for k in npz.files:
                arr = npz[k]
                if arr.ndim >= 1:
                    series_len = max(series_len, int(arr.shape[0]))

        X, y, _, _ = make_windows_from_npz([str(p)], ff, window_len=wl)
        y = np.asarray(y).reshape(-1)
        if scaler is not None:
            X = apply_scaler(X, scaler, clip=clip)

        probs = []
        with torch.no_grad():
            for xb, _ in batched_loader(X, y, batch_size=batch_size, shuffle=False):
                xb = torch.as_tensor(xb, device=device, dtype=torch.float32)
                logits = model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        prob = np.concatenate(probs, axis=0).reshape(-1)

        idx = np.arange(wl - 1, (wl - 1) + len(prob), dtype=np.int64)
        idx = idx[(idx >= 0)]
        if series_len == 0 and idx.size:
            series_len = int(idx.max()) + 1

        prob_full = np.full(series_len, np.nan, dtype=np.float32)
        pred_full = np.full(series_len, -1,   dtype=np.int8)

        n = min(len(prob), len(idx))
        if n == 0:
            out_paths.append(str(p))
            continue

        prob_full[idx[:n]] = prob[:n]
        pred_full[idx[:n]] = (prob[:n] >= thr).astype(np.int8)

        if in_place:
            out_path = p
        else:
            out_path = p.with_name(p.stem + suffix + p.suffix)

        save_dict = {k: np.array(npz[k]) for k in npz.files}
        save_dict["pred_threshold"] = np.array(thr, dtype=np.float32)
        LBL_KEY = "labels" if overwrite_labels and ("labels" in save_dict) else "labels_pred"
        save_dict[LBL_KEY] = pred_full.astype(np.int8)
        save_dict["pred_prob"] = prob_full.astype(np.float32)

        np.savez_compressed(out_path, **save_dict)
        out_paths.append(str(out_path))

    return out_paths