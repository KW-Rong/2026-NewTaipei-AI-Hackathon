# -*- coding: utf-8 -*-
"""YouBike 共用特徵工程 (SageMaker Processing Job 容器內執行)。

記憶體精簡 + 向量化版本 (適用 ml.m5.xlarge 16GB)：
- 逐檔讀取並立即轉精簡型別，避免全欄字串。
- lag / label 以『同站 + 精確位移時間戳』的 merge 產生，避免 per-row dict。
- 只用 1-4 月擬合特徵規格、類別、中位數；5/6 月僅套用。
- 標籤僅來自同站精確 t+30 / t+60；無對應則缺失並在該模型排除。
- 標籤天然限於各 split 月份 (df 僅含該 split)，不跨 split。
"""
import argparse
import gc
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

BASE = "/opt/ml/processing"
IN_TRAIN = os.path.join(BASE, "input", "train")
IN_VALID = os.path.join(BASE, "input", "validation")
IN_TEST = os.path.join(BASE, "input", "test")
OUT_DIR = os.path.join(BASE, "output")

REQUIRED_BASE_COLS = ["日期", "城市", "行政區", "場站名稱",
                      "總車柱數", "可借車數", "可還位數", "經度", "緯度"]
CLUSTERS = ["平穩低波動型", "均衡流動型", "雙尖峰高流動型", "通勤到達型"]
UNKNOWN_CLUSTER = "未知或新場站"
STATION_TYPES = ["捷運站", "公車站", "學校", "公園", "醫院", "商圈", "停車場"]
LAG_MINUTES = [30, 60, 90, 120]
HORIZONS = [30, 60]
MIN_MATCH_RATE = 0.60


def fail(msg, quality=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = quality or {}
    rep["fatal_error"] = msg
    with open(os.path.join(OUT_DIR, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print("[FATAL]", msg, file=sys.stderr)
    sys.exit(1)


def list_csvs(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".csv"))


def read_one_csv(path, want_extra):
    """讀單一 CSV，精簡型別。want_extra 決定是否保留 日曆/場站/分群 (train 才有)。"""
    usecols = list(REQUIRED_BASE_COLS)
    extra = ["日曆", "場站", "分群"]
    # 先讀表頭決定有哪些欄
    head = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    have_extra = [c for c in extra if c in head.columns]
    usecols += have_extra
    dtype = {
        "城市": "category", "行政區": "category", "場站名稱": "object",
        "總車柱數": "float32", "可借車數": "float32", "可還位數": "float32",
        "經度": "float32", "緯度": "float32",
    }
    for c in have_extra:
        dtype[c] = "category"
    df = pd.read_csv(path, usecols=usecols, dtype=dtype, encoding="utf-8-sig",
                     keep_default_na=False, na_values=[""])
    df["_dt"] = pd.to_datetime(df["日期"], errors="coerce")
    df.drop(columns=["日期"], inplace=True)
    return df, have_extra


def load_split(paths, split, expected_months, quality):
    if not paths:
        fail(f"{split}: 找不到任何 CSV", quality)
    frames, per_file, extra_cols = [], [], set()
    for p in paths:
        df, have_extra = read_one_csv(p, want_extra=(split == "train"))
        missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns and c != "日期"]
        if missing:
            fail(f"{split}/{os.path.basename(p)}: 缺少必要欄位 {missing}", quality)
        if int(df["_dt"].isna().sum()) > 0:
            fail(f"{split}/{os.path.basename(p)}: 有日期無法解析", quality)
        extra_cols.update(have_extra)
        per_file.append({"file": os.path.basename(p), "rows": int(len(df)),
                         "months": sorted(df["_dt"].dt.month.dropna().unique().tolist())})
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    got = sorted(full["_dt"].dt.month.dropna().unique().tolist())
    if set(got) != set(expected_months):
        fail(f"{split}: 月份與 split 不符，預期 {expected_months} 實際 {got}", quality)
    quality.setdefault("splits", {})[split] = {
        "files": per_file, "total_rows": int(len(full)), "months": got,
        "columns": list(full.columns)}
    return full, extra_cols


def normalize_and_dedup(df, split, quality):
    mins = df["_dt"].dt.minute
    offset_rate = float((~mins.isin([0, 30])).sum()) / max(1, len(df))
    quality.setdefault("time_offset_rate", {})[split] = round(offset_rate, 6)
    df["_dt"] = df["_dt"].dt.floor("30min")
    quality.setdefault("time_normalize_rule", {})[split] = (
        "floor_30min_low_offset" if offset_rate < 0.02 else "floor_30min_with_offset_warning")
    df["_skey"] = (df["城市"].astype(str) + "|" + df["行政區"].astype(str)
                   + "|" + df["場站名稱"].astype(str))
    before = len(df)
    df = df.sort_values(["_skey", "_dt"]).drop_duplicates(["_skey", "_dt"], keep="last")
    quality.setdefault("dedup", {})[split] = {"before": before, "after": int(len(df)),
                                              "removed": before - int(len(df))}
    return df.reset_index(drop=True)


def add_ratios(df):
    docks = df["總車柱數"].astype("float32").replace(0, np.nan)
    df["ratio_bike"] = (df["可借車數"] / docks).clip(0, 1).astype("float32")
    df["ratio_dock"] = (df["可還位數"] / docks).clip(0, 1).astype("float32")
    df["ratio_unavailable"] = (1.0 - (df["可借車數"] + df["可還位數"]) / docks).clip(0, 1).astype("float32")
    return df


def build_maps(train_df):
    cluster_map, type_map, calendar_map = {}, {}, {}
    if "分群" in train_df.columns:
        g = train_df.dropna(subset=["分群"])
        cluster_map = g.groupby("_skey")["分群"].agg(
            lambda s: s.value_counts().index[0]).astype(str).to_dict()
    if "場站" in train_df.columns:
        g = train_df.dropna(subset=["場站"])
        type_map = g.groupby("_skey")["場站"].agg(
            lambda s: s.value_counts().index[0]).astype(str).to_dict()
    if "日曆" in train_df.columns:
        tmp = train_df.dropna(subset=["日曆"]).copy()
        tmp["_d"] = tmp["_dt"].dt.strftime("%Y-%m-%d")
        calendar_map = tmp.groupby("_d")["日曆"].agg(
            lambda s: s.value_counts().index[0]).astype(str).to_dict()
    cal_categories = (sorted(train_df["日曆"].dropna().astype(str).unique().tolist())
                      if "日曆" in train_df.columns else [])
    return cluster_map, type_map, calendar_map, cal_categories


def apply_maps(df, cluster_map, type_map, calendar_map):
    c = df["_skey"].map(cluster_map)
    df["_cluster"] = c.where(c.isin(CLUSTERS), UNKNOWN_CLUSTER).fillna(UNKNOWN_CLUSTER)
    df["_types_str"] = df["_skey"].map(type_map).fillna("")
    for t in STATION_TYPES:
        df[f"type_{t}"] = df["_types_str"].str.split(",").apply(
            lambda lst: 1 if isinstance(lst, list) and t in lst else 0).astype("int8")
    if "日曆" in df.columns and df["日曆"].notna().any():
        df["_calendar"] = df["日曆"].astype(str).fillna("未知")
    else:
        dk = df["_dt"].dt.strftime("%Y-%m-%d")
        mapped = dk.map(calendar_map) if calendar_map else pd.Series([None] * len(df), index=df.index)
        dow = df["_dt"].dt.dayofweek
        fb = pd.Series(np.where(dow >= 5, "假日", "平日"), index=df.index)
        df["_calendar"] = mapped.fillna(fb)
    return df


def merge_shift(df, minutes, cols_src, cols_dst):
    """以 (_skey, _dt) 為鍵，取 t+minutes 的來源列值，對齊回目前列。
    等同：對每列 t，找同站在 (t - minutes) 的觀測 -> 用於 lag；
    或找 (t + minutes) 的觀測 -> 用於 label。呼叫端用 sign 控制方向。"""
    right = df[["_skey", "_dt"] + cols_src].copy()
    right["_dt"] = right["_dt"] - pd.Timedelta(minutes=minutes)  # 把來源時間往前挪，對齊到目前列
    right = right.rename(columns={s: d for s, d in zip(cols_src, cols_dst)})
    merged = df.merge(right, on=["_skey", "_dt"], how="left")
    return merged


def build_lags_labels(df, quality, split):
    """向量化：lag 讀過去(t-lag)，label 讀未來(t+H)。全用 merge。"""
    df = df.sort_values(["_skey", "_dt"]).reset_index(drop=True)
    base = df[["_skey", "_dt", "ratio_bike", "ratio_dock"]].copy()

    # lag: 目前列要 t-lag 的值 -> 把來源(其時間)往後移 lag 後對齊
    for lag in LAG_MINUTES:
        r = base.rename(columns={"ratio_bike": f"bike_lag_{lag}", "ratio_dock": f"dock_lag_{lag}"}).copy()
        r["_dt"] = r["_dt"] + pd.Timedelta(minutes=lag)
        df = df.merge(r[["_skey", "_dt", f"bike_lag_{lag}", f"dock_lag_{lag}"]],
                      on=["_skey", "_dt"], how="left")
    for h in [30, 60]:
        df[f"bike_delta_{h}"] = (df["ratio_bike"] - df[f"bike_lag_{h}"]).astype("float32")
        df[f"dock_delta_{h}"] = (df["ratio_dock"] - df[f"dock_lag_{h}"]).astype("float32")

    # 過去兩小時視窗 (t, t-30, t-60, t-90) 平均/標準差 + 連續性
    win_bike_cols = ["ratio_bike", "bike_lag_30", "bike_lag_60", "bike_lag_90"]
    win_dock_cols = ["ratio_dock", "dock_lag_30", "dock_lag_60", "dock_lag_90"]
    df["bike_2h_mean"] = df[win_bike_cols].mean(axis=1, skipna=True).astype("float32")
    df["bike_2h_std"] = df[win_bike_cols].std(axis=1, skipna=True).fillna(0.0).astype("float32")
    df["dock_2h_mean"] = df[win_dock_cols].mean(axis=1, skipna=True).astype("float32")
    df["dock_2h_std"] = df[win_dock_cols].std(axis=1, skipna=True).fillna(0.0).astype("float32")
    df["is_continuous"] = (df[win_bike_cols].notna().all(axis=1)).astype("int8")

    # labels: 目前列要 t+H 的值 -> 把來源時間往前移 H 後對齊
    label_avail = {}
    for h in HORIZONS:
        r = base.rename(columns={"ratio_bike": f"label_bike_{h}", "ratio_dock": f"label_dock_{h}"}).copy()
        r["_dt"] = r["_dt"] - pd.Timedelta(minutes=h)
        df = df.merge(r[["_skey", "_dt", f"label_bike_{h}", f"label_dock_{h}"]],
                      on=["_skey", "_dt"], how="left")
        label_avail[f"label_bike_{h}"] = int(df[f"label_bike_{h}"].notna().sum())
        label_avail[f"label_dock_{h}"] = int(df[f"label_dock_{h}"].notna().sum())
    quality.setdefault("label_availability", {})[split] = label_avail
    del base, r
    gc.collect()
    return df


def add_cyclical(df):
    dt = df["_dt"]
    hour = (dt.dt.hour + dt.dt.minute / 60.0).astype("float32")
    dow = dt.dt.dayofweek.astype("float32")
    month = dt.dt.month.astype("float32")
    df["cur_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["cur_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["cur_dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    df["cur_dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")
    df["cur_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12).astype("float32")
    df["cur_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12).astype("float32")
    df["cur_is_morning_peak"] = ((hour >= 7) & (hour < 9)).astype("int8")
    df["cur_is_evening_peak"] = ((hour >= 17) & (hour < 19)).astype("int8")
    for h in HORIZONS:
        tdt = dt + pd.Timedelta(minutes=h)
        th = (tdt.dt.hour + tdt.dt.minute / 60.0).astype("float32")
        tdow = tdt.dt.dayofweek.astype("float32")
        df[f"tgt_hour_sin_h{h}"] = np.sin(2 * np.pi * th / 24).astype("float32")
        df[f"tgt_hour_cos_h{h}"] = np.cos(2 * np.pi * th / 24).astype("float32")
        df[f"tgt_dow_sin_h{h}"] = np.sin(2 * np.pi * tdow / 7).astype("float32")
        df[f"tgt_dow_cos_h{h}"] = np.cos(2 * np.pi * tdow / 7).astype("float32")
        df[f"tgt_is_morning_peak_h{h}"] = ((th >= 7) & (th < 9)).astype("int8")
        df[f"tgt_is_evening_peak_h{h}"] = ((th >= 17) & (th < 19)).astype("int8")
    return df


def add_onehots(df, cal_categories):
    for cat in cal_categories:
        df[f"cal_{cat}"] = (df["_calendar"].astype(str) == cat).astype("int8")
    df["cal_未知"] = (~df["_calendar"].astype(str).isin(cal_categories)).astype("int8")
    for c in CLUSTERS:
        df[f"cluster_{c}"] = (df["_cluster"].astype(str) == c).astype("int8")
    df[f"cluster_{UNKNOWN_CLUSTER}"] = (df["_cluster"].astype(str) == UNKNOWN_CLUSTER).astype("int8")
    return df


def base_feature_columns(cal_categories):
    cols = ["總車柱數", "經度", "緯度", "ratio_bike", "ratio_dock", "ratio_unavailable"]
    for lag in LAG_MINUTES:
        cols += [f"bike_lag_{lag}", f"dock_lag_{lag}"]
    cols += ["bike_delta_30", "dock_delta_30", "bike_delta_60", "dock_delta_60",
             "bike_2h_mean", "bike_2h_std", "dock_2h_mean", "dock_2h_std", "is_continuous",
             "cur_hour_sin", "cur_hour_cos", "cur_dow_sin", "cur_dow_cos",
             "cur_month_sin", "cur_month_cos", "cur_is_morning_peak", "cur_is_evening_peak"]
    for cat in cal_categories:
        cols.append(f"cal_{cat}")
    cols.append("cal_未知")
    for t in STATION_TYPES:
        cols.append(f"type_{t}")
    for c in CLUSTERS:
        cols.append(f"cluster_{c}")
    cols.append(f"cluster_{UNKNOWN_CLUSTER}")
    return cols


def target_time_feature_names():
    return ["tgt_hour_sin", "tgt_hour_cos", "tgt_dow_sin", "tgt_dow_cos",
            "tgt_is_morning_peak", "tgt_is_evening_peak", "horizon_minutes"]


def fit_medians(train_df, cols):
    med = {}
    for c in cols:
        if c in train_df.columns:
            v = pd.to_numeric(train_df[c], errors="coerce").to_numpy()
            m = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
        else:
            m = 0.0
        med[c] = m if np.isfinite(m) else 0.0
    return med


def finalize(df, base_cols, medians, split, run_id):
    for h in HORIZONS:
        for k in ["bike", "dock"]:
            col = f"label_{k}_{h}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)
    tgt_cols = []
    for h in HORIZONS:
        tgt_cols += [f"tgt_hour_sin_h{h}", f"tgt_hour_cos_h{h}", f"tgt_dow_sin_h{h}",
                     f"tgt_dow_cos_h{h}", f"tgt_is_morning_peak_h{h}", f"tgt_is_evening_peak_h{h}"]
    for c in base_cols + tgt_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(medians.get(c, 0.0)).astype("float32")
    df = df.reset_index(drop=True)
    df["row_id"] = [f"{run_id}-{split}-{i}" for i in range(len(df))]
    df["station_key"] = df["_skey"].astype(str)
    df["cur_time"] = df["_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    keep = ["row_id", "station_key", "cur_time"] + base_cols + tgt_cols
    for h in HORIZONS:
        keep += [f"label_bike_{h}", f"label_dock_{h}"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    run_id = args.run_id
    os.makedirs(OUT_DIR, exist_ok=True)
    quality = {"run_id": run_id}
    try:
        train, _ = load_split(list_csvs(IN_TRAIN), "train", [1, 2, 3, 4], quality)
        valid, _ = load_split(list_csvs(IN_VALID), "validation", [5], quality)
        test, _ = load_split(list_csvs(IN_TEST), "test", [6], quality)

        train = normalize_and_dedup(train, "train", quality)
        valid = normalize_and_dedup(valid, "validation", quality)
        test = normalize_and_dedup(test, "test", quality)

        cluster_map, type_map, calendar_map, cal_categories = build_maps(train)
        quality["calendar_categories"] = cal_categories
        quality["n_train_stations"] = int(train["_skey"].nunique())

        train_keys = set(train["_skey"].unique())
        for nm, d in [("validation", valid), ("test", test)]:
            keys = d["_skey"].unique()
            matched = sum(1 for k in keys if k in train_keys)
            rate = matched / max(1, len(keys))
            quality.setdefault("station_match_rate", {})[nm] = {
                "stations": int(len(keys)), "matched": int(matched), "rate": round(rate, 4)}
            if rate < MIN_MATCH_RATE:
                fail(f"{nm}: 場站鍵對 1-4 月對應率 {rate:.2%} 低於下限 {MIN_MATCH_RATE:.0%}", quality)

        frames = {}
        for nm, d in [("train", train), ("validation", valid), ("test", test)]:
            d = add_ratios(d)
            d = apply_maps(d, cluster_map, type_map, calendar_map)
            d = build_lags_labels(d, quality, nm)
            d = add_cyclical(d)
            d = add_onehots(d, cal_categories)
            frames[nm] = d
            gc.collect()

        base_cols = base_feature_columns(cal_categories)
        med_cols = base_cols + [f"tgt_{s}_h{h}" for h in HORIZONS
                                for s in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                                          "is_morning_peak", "is_evening_peak"]]
        medians = fit_medians(frames["train"], med_cols)

        out = {}
        for nm in ["train", "validation", "test"]:
            out[nm] = finalize(frames[nm], base_cols, medians, nm, run_id)
            frames[nm] = None
            gc.collect()

        if not (list(out["train"].columns) == list(out["validation"].columns)
                == list(out["test"].columns)):
            fail("三個 split 欄位或順序不一致", quality)

        feat_cols = [c for c in out["train"].columns
                     if c not in ("row_id", "station_key", "cur_time") and not c.startswith("label_")]
        for nm in ["train", "validation", "test"]:
            arr = out[nm][feat_cols].to_numpy(dtype=np.float32)
            if not np.isfinite(arr).all():
                fail(f"{nm}: 特徵含 NaN 或 inf", quality)
            del arr

        out["train"].to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
        out["validation"].to_parquet(os.path.join(OUT_DIR, "validation.parquet"), index=False)
        out["test"].to_parquet(os.path.join(OUT_DIR, "test.parquet"), index=False)

        spec = {
            "run_id": run_id, "base_feature_order": base_cols,
            "target_time_features": target_time_feature_names(),
            "target_time_source_suffix": {str(h): f"_h{h}" for h in HORIZONS},
            "labels": {"bike-30": "label_bike_30", "dock-30": "label_dock_30",
                       "bike-60": "label_bike_60", "dock-60": "label_dock_60"},
            "clusters": CLUSTERS, "unknown_cluster": UNKNOWN_CLUSTER,
            "station_types": STATION_TYPES, "calendar_categories": cal_categories,
            "lag_minutes": LAG_MINUTES, "horizons": HORIZONS, "medians": medians,
            "n_base_features": len(base_cols),
            "n_features_per_model": len(base_cols) + len(target_time_feature_names())}
        with open(os.path.join(OUT_DIR, "feature_spec.json"), "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

        quality["row_counts"] = {nm: int(len(out[nm])) for nm in out}
        quality["n_base_features"] = len(base_cols)
        quality["n_features_per_model"] = len(base_cols) + len(target_time_feature_names())
        quality["label_nonnull"] = {
            nm: {lbl: int(pd.to_numeric(out[nm][lbl], errors="coerce").notna().sum())
                 for lbl in ["label_bike_30", "label_dock_30", "label_bike_60", "label_dock_60"]
                 if lbl in out[nm].columns}
            for nm in out}
        quality["status"] = "SUCCESS"
        with open(os.path.join(OUT_DIR, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(quality, f, ensure_ascii=False, indent=2, default=str)
        print("PROCESSING_DONE", quality["row_counts"])
    except SystemExit:
        raise
    except Exception as e:
        quality["traceback"] = traceback.format_exc()
        fail(f"未預期例外: {type(e).__name__}: {e}", quality)


if __name__ == "__main__":
    main()
