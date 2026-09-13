# -*- coding: utf-8 -*-
"""本機評估 (不建 Endpoint)：載入四個模型，對完整 5/6 月推論並產生指標與門檻。

流程：
1. 從 S3 下載四個模型工件 model.tar.gz (runs/{RUN_ID}/models/{target}/.../output/model.tar.gz)
   與 processed 的 validation.parquet、test.parquet、feature_spec.json。
2. 用 xgboost 載入 model.json，對 5 月(validation) 與 6 月(test) 各自推論。
3. 同 horizon 的 bike/dock 預測配對：先各自裁切 0~1；若兩者相加 >1 則等比例正規化。
4. 用完整 5 月選缺車/滿車警示門檻 (0.030~0.200 step 0.005)，排序依 F1 > Recall > Precision > 與0.10距離。
5. 門檻固定後才評估 6 月。輸出 MAE/RMSE/R²/Persistence baseline 與事件分類指標、混淆矩陣。
6. 分批輸出測試預測 Parquet (含 row_id/場站/現在時間/預測時間/真值) 到 runs/{RUN_ID}/eval/。
不寫入任何憑證。
"""
import io
import json
import os
import tarfile

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(ROOT, "config", "run_config.json")
WORK = os.path.join(ROOT, "artifacts", "eval_work")

TARGETS = ["bike-30", "dock-30", "bike-60", "dock-60"]


def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def clients(cfg):
    sess = boto3.Session(profile_name=cfg["aws"]["profile"], region_name=cfg["aws"]["region"])
    bc = Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 5})
    return sess, sess.client("s3", config=bc), sess.client("sagemaker", config=bc)


def download(s3, bucket, key, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file(bucket, key, dest)
    return dest


def find_model_artifact(sm, job_name):
    d = sm.describe_training_job(TrainingJobName=job_name)
    return d.get("ModelArtifacts", {}).get("S3ModelArtifacts")


def extract_model_json(tar_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getnames()
        tf.extractall(dest_dir)
    mj = os.path.join(dest_dir, "model.json")
    meta = os.path.join(dest_dir, "train_meta.json")
    return mj, (meta if os.path.exists(meta) else None), members


def build_matrix(df, spec, horizon):
    base = spec["base_feature_order"]
    suffix = spec["target_time_source_suffix"][str(horizon)]
    X = df[base].copy()
    tgt_map = {
        "tgt_hour_sin": f"tgt_hour_sin{suffix}", "tgt_hour_cos": f"tgt_hour_cos{suffix}",
        "tgt_dow_sin": f"tgt_dow_sin{suffix}", "tgt_dow_cos": f"tgt_dow_cos{suffix}",
        "tgt_is_morning_peak": f"tgt_is_morning_peak{suffix}",
        "tgt_is_evening_peak": f"tgt_is_evening_peak{suffix}",
    }
    for canon, src in tgt_map.items():
        X[canon] = df[src].to_numpy()
    X["horizon_minutes"] = horizon
    final_cols = base + spec["target_time_features"]
    return X[final_cols], final_cols


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(y_true))}


def persistence_baseline(y_true, cur_ratio):
    """baseline: 維持目前比例不變 (預測 = 現在比例)。"""
    return regression_metrics(y_true, cur_ratio)


def event_metrics(y_true, y_pred, thr, target_low=True):
    """低比例(缺車/滿車)事件：真值 <=0.10 為正例；預測 <=thr 判為警示。"""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    actual_pos = yt <= 0.10
    pred_pos = yp <= thr
    TP = int(np.sum(actual_pos & pred_pos))
    FP = int(np.sum(~actual_pos & pred_pos))
    FN = int(np.sum(actual_pos & ~pred_pos))
    TN = int(np.sum(~actual_pos & ~pred_pos))
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (TP + TN) / max(1, (TP + TN + FP + FN))
    return {"threshold": round(float(thr), 4), "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "TN": TN, "FP": FP, "FN": FN, "TP": TP}


def choose_threshold(y_true, y_pred, low, high, step, target_ratio=0.10):
    cands = np.round(np.arange(low, high + 1e-9, step), 4)
    best = None
    for thr in cands:
        m = event_metrics(y_true, y_pred, thr)
        key = (m["F1"], m["Recall"], m["Precision"], -abs(thr - target_ratio))
        if best is None or key > best[0]:
            best = (key, m)
    return best[1]


def pair_normalize(bike_pred, dock_pred):
    """同 horizon 的 bike/dock 預測：各自裁切 0~1，若相加 >1 則等比例正規化。"""
    b = np.clip(np.asarray(bike_pred, float), 0, 1)
    d = np.clip(np.asarray(dock_pred, float), 0, 1)
    s = b + d
    over = s > 1
    b = np.where(over, b / s, b)
    d = np.where(over, d / s, d)
    return b, d


def predict_split(models, spec, df):
    """對一個 split 產生四目標預測 (dict target->array)。分批以控制記憶體。"""
    preds = {}
    for target in TARGETS:
        horizon = int(target.split("-")[1])
        X, cols = build_matrix(df, spec, horizon)
        dm = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=cols)
        booster = models[target]
        bi = getattr(booster, "best_iteration", None)
        if bi is not None:
            preds[target] = booster.predict(dm, iteration_range=(0, bi + 1))
        else:
            preds[target] = booster.predict(dm)
    # 配對正規化 (30 一組, 60 一組)
    for h in [30, 60]:
        bk, dk = f"bike-{h}", f"dock-{h}"
        b, d = pair_normalize(preds[bk], preds[dk])
        preds[bk], preds[dk] = b, d
    return preds


def evaluate_split(split_name, df, preds, cfg):
    """回傳每個 target 的回歸+baseline 指標；分類指標由呼叫端用門檻算。"""
    out = {}
    for target in TARGETS:
        horizon = int(target.split("-")[1])
        kind = target.split("-")[0]  # bike / dock
        label = cfg["targets"][target]["label_col"]
        mask = pd.to_numeric(df[label], errors="coerce").notna().to_numpy()
        y = pd.to_numeric(df[label], errors="coerce").to_numpy()[mask]
        yp = preds[target][mask]
        cur = df[f"ratio_{kind}"].to_numpy()[mask]
        reg = regression_metrics(y, yp)
        base = persistence_baseline(y, cur)
        out[target] = {"regression": reg, "baseline_persistence": base,
                       "n_labeled": int(mask.sum())}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="training job 名稱後綴 (例如 r2)")
    args = ap.parse_args()
    job_suffix = f"-{args.suffix}" if args.suffix else ""
    cfg = load_cfg()
    sess, s3, sm = clients(cfg)
    bucket = cfg["aws"]["bucket"]
    s3c = cfg["s3"]
    os.makedirs(WORK, exist_ok=True)

    # 下載 processed 產物
    proc_local = os.path.join(WORK, "processed")
    for fn in ["validation.parquet", "test.parquet", "feature_spec.json"]:
        download(s3, bucket, s3c["processed_prefix"] + fn, os.path.join(proc_local, fn))
    with open(os.path.join(proc_local, "feature_spec.json"), "r", encoding="utf-8") as f:
        spec = json.load(f)
    val_df = pd.read_parquet(os.path.join(proc_local, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(proc_local, "test.parquet"))

    # 下載並載入四個模型
    models = {}
    model_uris = {}
    for target in TARGETS:
        job_name = f"{cfg['run_id']}-{target}{job_suffix}"
        art = find_model_artifact(sm, job_name)
        model_uris[target] = art
        assert art and art.startswith("s3://")
        _, rest = art.split("s3://", 1)
        b, key = rest.split("/", 1)
        tarp = os.path.join(WORK, target, "model.tar.gz")
        download(s3, b, key, tarp)
        mj, meta, members = extract_model_json(tarp, os.path.join(WORK, target))
        booster = xgb.Booster()
        booster.load_model(mj)
        # 還原 best_iteration (從 train_meta.json)
        if meta:
            with open(meta, "r", encoding="utf-8") as f:
                mm = json.load(f)
            bi = mm.get("best_iteration")
            if bi is not None:
                booster.best_iteration = int(bi)
        models[target] = booster

    # 推論
    val_preds = predict_split(models, spec, val_df)
    test_preds = predict_split(models, spec, test_df)

    # 回歸+baseline
    val_reg = evaluate_split("validation", val_df, val_preds, cfg)
    test_reg = evaluate_split("test", test_df, test_preds, cfg)

    # 5 月選門檻 (每個 target 各自)
    ts = cfg["threshold_search"]
    thresholds = {}
    val_event = {}
    for target in TARGETS:
        label = cfg["targets"][target]["label_col"]
        mask = pd.to_numeric(val_df[label], errors="coerce").notna().to_numpy()
        y = pd.to_numeric(val_df[label], errors="coerce").to_numpy()[mask]
        yp = val_preds[target][mask]
        chosen = choose_threshold(y, yp, ts["low"], ts["high"], ts["step"], ts["target_ratio"])
        thresholds[target] = chosen["threshold"]
        val_event[target] = chosen

    # 6 月固定門檻評估
    test_event = {}
    for target in TARGETS:
        label = cfg["targets"][target]["label_col"]
        mask = pd.to_numeric(test_df[label], errors="coerce").notna().to_numpy()
        y = pd.to_numeric(test_df[label], errors="coerce").to_numpy()[mask]
        yp = test_preds[target][mask]
        test_event[target] = event_metrics(y, yp, thresholds[target])

    # 輸出測試預測 Parquet (含追溯欄)
    pred_rows = test_df[["row_id", "station_key", "cur_time"]].copy()
    for target in TARGETS:
        horizon = int(target.split("-")[1])
        pred_rows[f"pred_{target}"] = test_preds[target]
        label = cfg["targets"][target]["label_col"]
        pred_rows[f"true_{target}"] = pd.to_numeric(test_df[label], errors="coerce").to_numpy()
    # 預測時間 = 現在 + horizon (以 30/60 兩種，附兩欄)
    cur_dt = pd.to_datetime(test_df["cur_time"])
    pred_rows["pred_time_30"] = (cur_dt + pd.Timedelta(minutes=30)).dt.strftime("%Y-%m-%d %H:%M:%S")
    pred_rows["pred_time_60"] = (cur_dt + pd.Timedelta(minutes=60)).dt.strftime("%Y-%m-%d %H:%M:%S")

    eval_local = os.path.join(WORK, "eval_out")
    os.makedirs(eval_local, exist_ok=True)
    pred_path = os.path.join(eval_local, "test_predictions.parquet")
    pred_rows.to_parquet(pred_path, index=False)

    report = {
        "run_id": cfg["run_id"],
        "model_artifacts": model_uris,
        "thresholds": thresholds,
        "validation_may": {"regression": val_reg, "event": val_event},
        "test_june": {"regression": test_reg, "event": test_event},
        "n_val": int(len(val_df)), "n_test": int(len(test_df)),
    }
    report_path = os.path.join(eval_local, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 上傳 eval 輸出到 runs/{RUN_ID}/eval/
    for fn in ["test_predictions.parquet", "evaluation_report.json"]:
        s3.upload_file(os.path.join(eval_local, fn), bucket, s3c["eval_prefix"] + fn,
                       ExtraArgs={"ServerSideEncryption": "AES256"})

    # 本機也存一份報告 (不含憑證)
    with open(os.path.join(ROOT, "artifacts", "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
