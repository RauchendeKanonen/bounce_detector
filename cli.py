from bounce_detector.config import load_config_json, resolve_run_config, merge_dotset
from bounce_detector.model import TinyTemporalCNN
from bounce_detector.train import train_from_npz, train_from_npz_streaming
from bounce_detector.checkpoint import save_trained, load_trained
from bounce_detector.evaluate import evaluate_from_npz, best_threshold_on_val
from bounce_detector.predict import predict_from_npz_writeback
from bounce_detector.search_algo import run_search_optuna
import argparse
import glob
import torch
import json
from pathlib import Path
from dataclasses import dataclass, asdict, is_dataclass
from bounce_detector.config import RunConfig, RunScore

def expand_globs(paths_or_globs: list[str] | None) -> list[str]:
    if not paths_or_globs:
        return []
    out = []
    for p in paths_or_globs:
        out.extend(sorted(glob.glob(p)))
    return out or (paths_or_globs or [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, save, evaluate, and predict for Bounce Detector (modular).")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config with data/model/train/eval blocks.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config keys: key.sub=value (repeatable)")
    parser.add_argument("--out", required=True, help="Path to save model (e.g. models/bounce.ckpt)")
    parser.add_argument("--device", default=None, help="cuda|cpu|auto (overrides train.device)")
    parser.add_argument("--score-test", action="store_true", help="After training, score the test set")
    parser.add_argument("--predict-back", action="store_true", help="Run inference on test files and write predictions back (.pred.npz)\n "
                                                                    "--predict-back --config run_config.json --out /tmp/out --checkpoint /tmp/out/trial_0001/model.ckpt")
    parser.add_argument("--predict-in-place", action="store_true", help="Overwrite the input NPZ file(s) (dangerous)")
    parser.add_argument("--predict-overwrite-labels", action="store_true", help="Write predicted labels into existing 'labels'")
    parser.add_argument("--predict-batch", type=int, default=512, help="Batch size during prediction.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained checkpoint (.ckpt/.pkl). Required for predict-back unless --out is a checkpoint file.")
    parser.add_argument("--only-valid_idx", type=lambda x: str(x).lower() in ('1','true','yes'), default=False, help="Predict only on valid_idx positions.")


    #################### Search #####################
    parser.add_argument("--search", type=int, default=0, help="Number of Optuna trials (0 = no search)\n "
                                                              "--search 10 --study-name test_study --config run_config.json --out ./search_runs")
    parser.add_argument("--study-name", type=str, default="bounce_search")
    parser.add_argument("--storage", type=str, default=None, help='Optuna storage, e.g. sqlite:///bounce.db')


    args = parser.parse_args()

    cfg_dict = load_config_json(args.config)
    for ov in args.overrides or []:
        if "=" not in ov:
            print(f"[WARN] Ignoring malformed override: {ov}")
            continue
        k, v = ov.split("=", 1)
        merge_dotset(cfg_dict, k.strip(), v.strip())

    if args.device:
        cfg_dict.setdefault("train", {})["device"] = args.device

    rc = resolve_run_config(cfg_dict)

    train_files = expand_globs(rc.data.train_glob)
    val_files   = expand_globs(rc.data.val_glob)
    test_files  = expand_globs(rc.data.test_glob)
    if not train_files or not val_files or not test_files:
        raise SystemExit("Train/Val/Test files must be specified in config.")

    feature_fields = rc.data.feature_fields or []
    if not feature_fields:
        raise SystemExit("feature_fields must be provided in config.data.feature_fields.")

    device = rc.train.device if rc.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTemporalCNN(
        F=len(feature_fields), T=rc.data.window_len,
        hidden=rc.model.hidden, n_out=rc.model.out_dim,
        k=rc.model.k, layers=rc.model.layers, dropout=rc.model.dropout
    )

    # Prediction-only shortcut
    if args.predict_back or args.predict_in_place:
        ckpt = args.checkpoint
        if ckpt is None:
            p_out = Path(args.out)
            if p_out.is_file() and p_out.suffix in {".ckpt", ".pkl", ".pickle"}:
                ckpt = str(p_out)
            else:
                raise SystemExit("For --predict-back, provide --checkpoint path/to.ckpt (or set --out to a checkpoint file).")
        payload, _, _ = load_trained(ckpt, build_model=True, device="auto")
        try:
            data = json.loads((Path(ckpt).parent / "run_score.json").read_text())
            # recreate dataclass
            Rs = RunScore(**data)
            print(f"run_score.json loaded")
            payload["threshold"] = Rs.threshold
        except:
            pass

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
        exit(0)


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



    # Train
    # trained = train_from_npz(
    #     train_paths=train_files[:1],
    #     val_paths=val_files[:1],
    #     feature_fields=feature_fields,
    #     window_len=rc.data.window_len,
    #     epochs=rc.train.epochs,
    #     batch_size=rc.train.batch_size,
    #     lr=rc.train.lr,
    #     weight_decay=rc.train.weight_decay,
    #     device=device,
    #     class_weight=rc.train.class_weight_pos,
    #     precision16=rc.train.precision16,
    #     model=model,
    #     scaler_clip=rc.data.scaler_clip,
    #     early_stopping_patience=rc.train.early_stopping_patience,
    # )

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

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_trained(trained, out_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Evaluate
    payload = load_trained(str(ckpt_path))
    metrics = evaluate_from_npz(payload, test_files, feature_fields)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.score_test:
        best_thr, thr_info = best_threshold_on_val(payload, val_files, feature_fields)
        payload["threshold"] = best_thr
        print("Best threshold on val:", best_thr, "info:", thr_info)
