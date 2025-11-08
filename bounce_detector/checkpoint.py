from __future__ import annotations
from pathlib import Path
import pickle, json, torch, numpy as np
from .model import TinyTemporalCNN
from .data import apply_scaler, ffill_bfill_time_batch
from typing import List
from dataclasses import dataclass

_REQUIRED_KEYS = {
    "model_class", "model_state", "model_init_kwargs",
    "feature_fields", "window_len", "scaler"
}

def save_trained(trained: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [k for k in _REQUIRED_KEYS if k not in trained]
    if missing:
        raise ValueError(f"save_trained: payload missing keys: {missing}")
    trained.setdefault("run_config", trained.get("run_config", {
        "version": trained.get("version", "v1"),
        "data": {"feature_fields": trained.get("feature_fields", []),
                 "window_len": trained.get("window_len", None)}
    }))
    with open(path, "wb") as f:
        pickle.dump(trained, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        pass
    return path

def _ensure_model_kwargs(payload: dict) -> dict:
    mk = dict(payload.get("model_init_kwargs") or {})
    if "F" not in mk:
        ff = payload.get("feature_fields", [])
        mk["F"] = int(len(ff)) if ff else int(payload.get("F", 0))
    if "T" not in mk:
        mk["T"] = int(payload.get("window_len"))
    mk.setdefault("hidden", 64)
    mk.setdefault("n_out", 1)
    mk.setdefault("k", 9)
    mk.setdefault("layers", 2)
    mk.setdefault("dropout", 0.1)
    if not mk.get("F") or not mk.get("T"):
        raise ValueError("model_init_kwargs requires F and T; could not infer from payload.")
    return mk

def load_trained(path: str | Path, build_model: bool = False, device: str = "auto"):
    path = Path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    payload = dict(payload)
    payload.setdefault("feature_fields", payload.get("feature_fields", []))
    payload.setdefault("run_config", payload.get("run_config", {}))
    payload.setdefault("version", payload.get("version", payload["run_config"].get("version", "v1")))
    payload.setdefault("threshold", payload.get("threshold", 0.5))
    payload.setdefault("scaler_clip", payload.get("scaler_clip", 8.0))
    if "window_len" not in payload and "T" in payload:
        payload["window_len"] = int(payload["T"])
    payload["model_init_kwargs"] = _ensure_model_kwargs(payload)
    if not build_model:
        return payload
    mk = payload["model_init_kwargs"]
    model = TinyTemporalCNN(**mk)
    dev = torch.device("cuda" if (device in (None, "auto") and torch.cuda.is_available()) else (device or "cpu"))
    model.to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return payload, model, dev

class BounceDetector:
    def __init__(self, model: TinyTemporalCNN, feature_fields: List[str], window_len: int, scaler, device: str | None = None):
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
        import torch
        return torch.from_numpy(X).float().to(self.device)

    @torch.no_grad()
    def predict_window(self, window_np: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            if window_np.ndim == 3:
                if window_np.shape[1] != 1 and window_np.shape[2] == 1:
                    import numpy as np
                    window_np = np.transpose(window_np, (0, 2, 1))
                window_np = ffill_bfill_time_batch(window_np[None, ...])[0]
            elif window_np.ndim == 4:
                import numpy as np
                if window_np.shape[2] != 1 and window_np.shape[3] == 1:
                    window_np = np.transpose(window_np, (0, 1, 3, 2))
                window_np = ffill_bfill_time_batch(window_np)
            window_np = apply_scaler(window_np, self.scaler, clip=8.0)
        else:
            import numpy as np
            window_np = np.nan_to_num(window_np, nan=0.0, posinf=8.0, neginf=-8.0)
        import torch
        x = self._to_tensor(window_np)
        logits = self.model(x).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

def create_bouncedetector(path: str, device: str | None = None) -> BounceDetector:
    payload, model, dev = load_trained(path, build_model=True, device=("auto" if device is None else device))
    ff = payload.get("feature_fields", [])
    mk = payload.get("model_init_kwargs", {})
    if mk and "F" in mk and len(ff) and mk["F"] != len(ff):
        print(f"[WARN] feature_fields length ({len(ff)}) != model F ({mk['F']}). Check the checkpoint or feature configuration.", flush=True)
    det = BounceDetector(
        model=model,
        feature_fields=ff,
        window_len=payload.get("window_len"),
        device=str(dev),
        scaler=payload.get("scaler", None),
    )
    setattr(det, "threshold", float(payload.get("threshold", 0.5)))
    setattr(det, "scaler_clip", float(payload.get("scaler_clip", 8.0)))
    setattr(det, "run_config", payload.get("run_config", {}))
    setattr(det, "model_init_kwargs", mk)
    return det
