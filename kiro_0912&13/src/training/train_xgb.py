# -*- coding: utf-8 -*-
"""YouBike XGBoost Script Mode 訓練腳本 (SageMaker Training 容器內執行)。

單一模型訓練：依 --target 選定標籤 (bike-30 / dock-30 / bike-60 / dock-60)，
讀共用 train/validation Parquet 與 feature_spec.json，組裝該 horizon 的特徵矩陣，
套用樣本權重 (目標<=0.10 權重 3，其餘 1)，以 5 月做 early stopping。
輸出 model.json、best_iteration、使用的 feature_spec 副本到 /opt/ml/model。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def build_matrix(df, spec, horizon):
    """依 feature_spec 組裝特徵矩陣：base 特徵 + 該 horizon 的 tgt_* (由 _h{H} 重新命名) + horizon_minutes。"""
    base = spec["base_feature_order"]
    suffix = spec["target_time_source_suffix"][str(horizon)]  # e.g. "_h30"
    X = df[base].copy()
    # tgt_* 對應欄
    tgt_map = {
        "tgt_hour_sin": f"tgt_hour_sin{suffix}",
        "tgt_hour_cos": f"tgt_hour_cos{suffix}",
        "tgt_dow_sin": f"tgt_dow_sin{suffix}",
        "tgt_dow_cos": f"tgt_dow_cos{suffix}",
        "tgt_is_morning_peak": f"tgt_is_morning_peak{suffix}",
        "tgt_is_evening_peak": f"tgt_is_evening_peak{suffix}",
    }
    for canon, src in tgt_map.items():
        X[canon] = df[src].to_numpy()
    X["horizon_minutes"] = horizon
    # 最終欄位順序 = base + target_time_features
    final_cols = base + spec["target_time_features"]
    X = X[final_cols]
    return X, final_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)          # bike-30 / dock-30 / bike-60 / dock-60
    ap.add_argument("--label-col", required=True)        # label_bike_30 ...
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--num-round", type=int, default=1200)
    ap.add_argument("--eta", type=float, default=0.045)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-child-weight", type=float, default=8)
    ap.add_argument("--subsample", type=float, default=0.82)
    ap.add_argument("--colsample-bytree", type=float, default=0.82)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=2)
    ap.add_argument("--max-bin", type=int, default=192)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--low-ratio-threshold", type=float, default=0.10)
    ap.add_argument("--low-ratio-weight", type=float, default=3.0)
    args = ap.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    valid_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    spec_dir = os.environ.get("SM_CHANNEL_SPEC", "/opt/ml/input/data/spec")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(spec_dir, "feature_spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)

    train_df = pd.read_parquet(os.path.join(train_dir, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(valid_dir, "validation.parquet"))

    label = args.label_col
    # 只保留該標籤非缺失的樣本 (無對應 t+H 者排除)
    train_df = train_df[pd.to_numeric(train_df[label], errors="coerce").notna()].reset_index(drop=True)
    valid_df = valid_df[pd.to_numeric(valid_df[label], errors="coerce").notna()].reset_index(drop=True)

    Xtr, final_cols = build_matrix(train_df, spec, args.horizon)
    Xva, _ = build_matrix(valid_df, spec, args.horizon)
    ytr = pd.to_numeric(train_df[label], errors="coerce").clip(0, 1).to_numpy()
    yva = pd.to_numeric(valid_df[label], errors="coerce").clip(0, 1).to_numpy()

    # 樣本權重：目標 <=0.10 -> 權重 3，其餘 1
    wtr = np.where(ytr <= args.low_ratio_threshold, args.low_ratio_weight, 1.0)

    dtrain = xgb.DMatrix(Xtr.to_numpy(dtype=np.float32), label=ytr, weight=wtr,
                         feature_names=final_cols)
    dvalid = xgb.DMatrix(Xva.to_numpy(dtype=np.float32), label=yva, feature_names=final_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "alpha": args.alpha,
        "lambda": args.reg_lambda,
        "max_bin": args.max_bin,
        "seed": args.seed,
    }

    evals_result = {}
    booster = xgb.train(
        params, dtrain, num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result, verbose_eval=50,
    )

    best_iter = int(getattr(booster, "best_iteration", args.num_round - 1))
    best_score = float(getattr(booster, "best_score", float("nan")))

    # 輸出模型與中繼資料
    booster.save_model(os.path.join(model_dir, "model.json"))
    with open(os.path.join(model_dir, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    meta = {
        "target": args.target, "label_col": label, "horizon": args.horizon,
        "best_iteration": best_iter, "best_validation_rmse": best_score,
        "n_train": int(len(ytr)), "n_valid": int(len(yva)),
        "feature_order": final_cols, "n_features": len(final_cols),
        "params": params, "num_round": args.num_round,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    with open(os.path.join(model_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # 訓練紀錄
    with open(os.path.join(model_dir, "evals_result.json"), "w", encoding="utf-8") as f:
        json.dump(evals_result, f, ensure_ascii=False, indent=2)
    print(f"TRAIN_DONE target={args.target} best_iter={best_iter} best_rmse={best_score:.6f} "
          f"n_train={len(ytr)} n_valid={len(yva)} n_feat={len(final_cols)}")


if __name__ == "__main__":
    main()
