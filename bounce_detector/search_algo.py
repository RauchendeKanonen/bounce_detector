from __future__ import annotations
from copy import deepcopy

from .predict import predict_from_npz
from .config import RunConfig, RunScore
import numpy as np
import torch
import json
from pathlib import Path
from .data import make_windows_from_npz, apply_scaler, load_one_npz
from .model import TinyTemporalCNN
from .train import batched_loader, _device_from_cfg, train_from_npz, train_from_npz_streaming
from .evaluate import evaluate_from_npz
from .config import asdict_dc, DataConfig, ModelConfig, TrainConfig, EvalConfig
from .checkpoint import save_trained
from dataclasses import dataclass, asdict, is_dataclass



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

def _sample_search_space(trial, rc: RunConfig):
    """Define a compact, sane search space."""
    space = {}
    # Data
    space["window_len"] = trial.suggest_categorical("data.window_len", [1024])
    # Model
    space["hidden"]  = trial.suggest_categorical("model.hidden", [4,8,16])
    space["k"]       = trial.suggest_categorical("model.k", [9,11,13,15])
    space["layers"]  = trial.suggest_categorical("model.layers", [2,3,4,5,6])
    space["dropout"] = trial.suggest_float("model.dropout", 0.2, 0.5, step=0.05)
    # Train
    space["lr"]           = trial.suggest_float("train.lr", 0.0001, 0.0003, log=True)
    space["weight_decay"] = trial.suggest_float("train.weight_decay", 1e-6, 4*1e-6, log=True)
    space["batch_size"]   = trial.suggest_categorical("train.batch_size", [1024])
    return space

def _resolve_trial_configs(base: RunConfig, space: dict) -> RunConfig:
    """Create a shallow copy of base RunConfig with sampled overrides."""
    # copy dataclasses
    data = deepcopy(asdict_dc(base.data));  model = deepcopy(asdict_dc(base.model))
    train = deepcopy(asdict_dc(base.train)); evalc = deepcopy(asdict_dc(base.eval) if base.eval else {})
    # apply overrides
    data["window_len"] = space["window_len"]
    model.update(dict(hidden=space["hidden"], k=space["k"], layers=space["layers"], dropout=space["dropout"], window_len=space["window_len"]))
    train.update(dict(lr=space["lr"], weight_decay=space["weight_decay"], batch_size=space["batch_size"]))
    # pack back
    return RunConfig(data=DataConfig(**data), model=ModelConfig(**model), train=TrainConfig(**train), eval=EvalConfig(**evalc) if evalc else None, version=base.version, git_commit=base.git_commit)



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

        globals()['_LAST_RESOLVED_RUNCONFIG_DICT'] = asdict_dc(rc)  # embed in payload
        device = "cuda"#_device_from_cfg(rc.train)

        trained = train_from_npz_streaming(
            train_paths=train_files,  # many files ok
            val_paths=val_files,  # many files ok
            feature_fields=rc.data.feature_fields,
            window_len=rc.data.window_len,
            epochs=rc.train.epochs,
            batch_size=rc.train.batch_size,
            lr=rc.train.lr,
            weight_decay=rc.train.weight_decay,
            device=device,
            class_weight="auto",  # or a float
            precision16=rc.train.precision16,
            model=model,
            scaler_clip=rc.data.scaler_clip,
            early_stopping_patience=rc.train.early_stopping_patience,
        )

        custom_score, ones_score, changes_score, price_score, threshold, thres_std_dev = predicted_score(trained,
                                                                               test_files,
                                                                               rc.data.feature_fields)


        # Pick threshold on validation and record best AUPRC
        best_thr, thr_info = _best_threshold_on_val(trained, val_files, rc.data.feature_fields)
        trained["threshold"] = best_thr

        # Save trial artifacts
        tdir = out_dir / f"trial_{trial.number:04d}"
        tdir.mkdir(parents=True, exist_ok=True)
        save_trained(trained, tdir / "model.ckpt")
        (tdir / "search_space.json").write_text(json.dumps(space, indent=2))
        (tdir / "run_config.json").write_text(json.dumps(asdict_dc(rc), indent=2))



        # Report to Optuna (so pruner can act if you later add intermediate reports)
        score = float(thr_info.get("auprc", float("nan")))

        Rs = RunScore(custom_score=custom_score,
                      ones_score=ones_score,
                      changes_score=changes_score,
                      price_score=price_score,
                      auprc_score=score,
                      threshold = threshold,
                      thres_std_dev= thres_std_dev)
        (tdir / "run_score.json").write_text(json.dumps(asdict_dc(Rs), indent=2))

        trial.set_user_attr("best_threshold", best_thr)
        trial.set_user_attr("val_auprc", score)
        return custom_score

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


def moving_average(x, window: int):
    """Safe moving average that handles small/large windows gracefully."""
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window <= 1:
        return x.copy()
    if window > len(x):
        # Fallback: use the global mean and return a flat series
        return np.full_like(x, x.mean(), dtype=float)
    w = np.ones(window, dtype=float) / window
    y = np.convolve(x, w, mode='valid')
    # pad with the first valid average to keep length
    pad = np.full(window - 1, y[0], dtype=float)
    return np.concatenate((pad, y))


def _custom_score_once_with_range(trained: dict, y_pred: np.ndarray, mid: np.ndarray,
                                  min_price: float, max_price: float):
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    mid = np.asarray(mid, dtype=float).reshape(-1)

    pad_len = int(trained["window_len"]) - 1
    y_pred_padded = np.pad(y_pred, (pad_len, 0), mode='constant') if pad_len > 0 else y_pred
    n = len(y_pred_padded)
    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0

    changes = int(np.sum(y_pred_padded[1:] != y_pred_padded[:-1]))
    ones = int(np.sum(y_pred_padded[1:] > 0))

    changes_score = float(np.clip(1.0 - (changes / n), 0.0, 1.0))
    ones_score    = float(np.clip(1.0 - (ones    / n), 0.0, 1.0))

    denom = (max_price - min_price) if max_price > min_price else 1.0
    chosen_prices = mid[y_pred_padded.astype(bool)]
    avg_price = float(np.mean(chosen_prices)) if chosen_prices.size else max_price
    price_score = float(np.clip(1.0 - ((avg_price - min_price) / denom), 0.0, 1.0))

    composite = (price_score + changes_score + ones_score) / 3.0
    return price_score, changes_score, ones_score, composite

def optimize_threshold(trained: dict, probs: np.ndarray, mid: np.ndarray):
    """
    Vectorized sweep over thresholds in [0.00, 1.00] step 0.01.
    Aligns predictions to 'mid' by LEFT padding/truncation so mask length == len(mid).
    Price score uses the *achievable* future price: forward rolling MIN over a lookahead horizon.
    Returns (best_thr, (price_score, changes_score, ones_score, composite)).
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)  # N
    mid   = np.asarray(mid,   dtype=float).reshape(-1)  # L
    if probs.size == 0 or mid.size == 0:
        return 0.5, (0.0, 0.0, 0.0, 0.0)

    L = mid.shape[0]
    N = probs.shape[0]
    w = int(trained.get("window_len", 1))
    lookahead = int(trained.get("price_lookahead", w))
    lookahead = max(1, lookahead)

    # Build threshold grid and unpadded predictions: (N, G)
    grid = np.linspace(0.0, 1.0, 101)
    Y = (probs[:, None] >= grid[None, :]).astype(np.int8)  # (N, G)

    # Align to length L by LEFT padding (or truncate if ever longer)
    actual_pad = L - N
    if actual_pad > 0:
        pad = np.zeros((actual_pad, Y.shape[1]), dtype=np.int8)
        Yp = np.vstack([pad, Y])                       # (L, G)
    elif actual_pad < 0:
        Yp = Y[-L:, :]                                 # (L, G)
    else:
        Yp = Y                                         # (L, G)

    # --- Precompute smoothed band and achievable future price ---
    # (1) Smoothed band for normalization
    avg_mid = moving_average(mid, w)
    min_price = float(np.min(avg_mid))
    max_price = float(np.max(avg_mid))
    denom = (max_price - min_price) if max_price > min_price else 1.0

    # (2) Forward rolling minimum over 'lookahead' (vectorized)
    # future_min[i] = min(mid[i : i+lookahead])
    # Build stacked shifts and take columnwise min
    # Shape: (lookahead, L)
    if lookahead == 1:
        future_min = mid.copy()
    else:
        rows = []
        # Row k contains mid shifted by k to the left, padded with +inf on the right
        # so each column j aggregates mid[j : j+lookahead] across rows.
        inf = float("inf")
        for k in range(lookahead):
            if k == 0:
                rows.append(mid)
            else:
                rows.append(np.pad(mid[k:], (0, k), constant_values=inf))
        future_min = np.vstack(rows).min(axis=0)  # (L,)

    # --- Metrics using padded mask Yp (length L) ---
    # Changes / ones
    changes = np.sum(np.diff(Yp, axis=0) != 0, axis=0)  # (G,)
    ones    = np.sum(Yp[1:, :], axis=0)                 # (G,)
    changes_score = 1.0 - (changes / L)
    ones_score    = 1.0 - (ones    / L)

    # Price score based on achievable future price at signal indices
    # avg_achievable(thr) = mean(future_min at positions where mask==1)
    # Implement via weighted average with fallback when count==0
    sums   = (future_min[:, None] * Yp).sum(axis=0)     # (G,)
    counts = Yp.sum(axis=0)                             # (G,)
    avg_achievable = np.divide(
        sums, counts,
        out=np.full_like(sums, max_price, dtype=float),
        where=counts > 0
    )
    price_score = 1.0 - (avg_achievable - min_price) / denom

    # Clamp and composite
    price_score   = np.clip(price_score,   0.0, 1.0)
    changes_score = np.clip(changes_score, 0.0, 1.0)
    ones_score    = np.clip(ones_score,    0.0, 1.0)

    # Composite (match your evaluator; swap to weighted if you prefer)
    comp = (price_score + changes_score + ones_score) / 3.0
    # For weighted variant:
    # comp = (2.0 * price_score + changes_score + ones_score) / 4.0

    best_idx = int(np.argmax(comp))
    best_thr = float(grid[best_idx])
    best_comp = (
        float(price_score[best_idx]),
        float(changes_score[best_idx]),
        float(ones_score[best_idx]),
        float(comp[best_idx]),
    )
    return best_thr, best_comp



def predicted_score(trained: dict, test_files: list[str], feature_fields: list[str]):
    """
    Evaluate the first up-to-4 test files. For each file:
    - compute probs once
    - search threshold
    - use the single computed best-score breakdown (no re-scoring)
    Then average across files with safe means.
    """
    price_scores = []
    changes_scores = []
    ones_scores = []
    thresholds = []

    for p in test_files:
        prediction_set = predict_from_npz(trained, [p])
        probs = np.asarray(prediction_set["prob"]).reshape(-1)

        data = load_one_npz(p)
        mid = np.asarray(data["sampled"]["mid"]).reshape(-1)

        thr, (price_s, changes_s, ones_s, comp) = optimize_threshold(trained, probs, mid)

        # Keep the single, consistent set of metrics computed at best thr
        thresholds.append(thr)
        price_scores.append(price_s)
        changes_scores.append(changes_s)
        ones_scores.append(ones_s)

    def _safe_mean(a, default=0.0):
        return float(np.mean(a)) if len(a) else default

    price_score = _safe_mean(price_scores)
    changes_score = _safe_mean(changes_scores)
    ones_score = _safe_mean(ones_scores)


    thr_mean = _safe_mean(thresholds, default=0.5)
    thr_std = float(np.std(thresholds)) if len(thresholds) else 0.0

    custom = (2 * price_score + changes_score + ones_score + (1.0-thr_std)) / 5.0

    return custom, ones_score, changes_score, price_score, thr_mean, thr_std
