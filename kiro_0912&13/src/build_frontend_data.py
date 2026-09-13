# -*- coding: utf-8 -*-
"""後端資料建構器：把本次四個 SageMaker 模型的『可追溯回放預測』整理成前端資料集。

資料來源（全部本地、可追溯本次 RUN_ID 四個模型工件）：
  - artifacts/eval_work/processed/test.parquet   ← 站況特徵（容量/座標/當下比例/分群/型別）
  - artifacts/fe_work/test_predictions.parquet   ← 四模型輸出比例 pred_bike-30/dock-30/bike-60/dock-60
  - artifacts/evaluation_report.json             ← 5 月校正的四向警示門檻 + 6 月評估指標
  - config/run_config.json                       ← RUN_ID、分群名稱、模型設定

核心原則（對應使用者需求）：
1. 模型直接輸出的是『比例』；後端以『總車柱數』換算成車數／空位數，並限制不合理值、
   處理 bike+dock 相加超過容量的情況。
2. 缺失／停用／資料不足（is_continuous=0 或 lag 缺）不當成 0，明確標記 dataStatus。
3. 分群直接由 processed 的 cluster one-hot 還原；未知或新場站 → 前端顯示「未分群新站」，
   不硬分進四群。
4. 只輸出一天（預設 2026-06-29）作為『歷史回放』；並記錄 replayDate / baseTime / 模型版本。

輸出：
  - frontend/appData.js       （相容既有 schema：window.appData + window.allStations）
  - frontend/model_meta.json  （門檻、指標、模型版本、來源，供前端顯示與稽核）

不含任何 AWS 憑證；不呼叫 AWS。純本地檔案處理。
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")
CONFIG_PATH = os.path.join(ROOT, "config", "run_config.json")
FRONTEND = os.path.join(ROOT, "frontend")

PROC_CANDIDATES = [
    os.path.join(ART, "eval_work", "processed", "test.parquet"),
    os.path.join(ART, "verify_work", "test.parquet"),
]
PREDS_PATH = os.path.join(ART, "fe_work", "test_predictions.parquet")
EVAL_REPORT = os.path.join(ART, "evaluation_report.json")

CLUSTER_DISPLAY = {
    "平穩低波動型": "平穩低波動型",
    "均衡流動型": "均衡流動型",
    "雙尖峰高流動型": "雙尖峰高流動型",
    "通勤到達型": "通勤到達型",
    "未知或新場站": "未分群新站",
}
CLUSTER_ORDER = ["平穩低波動型", "均衡流動型", "雙尖峰高流動型", "通勤到達型"]
UNKNOWN_KEY = "未知或新場站"
UNKNOWN_DISPLAY = "未分群新站"
STATION_TYPES = ["捷運站", "公車站", "學校", "公園", "醫院", "商圈", "停車場"]

SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]  # 48 格
EVENT_RATIO = 0.10  # 事件定義：可借/可還比例 <= 0.10 視為缺車/滿站


def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_processed():
    for p in PROC_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到 processed test.parquet；預期於 " + " 或 ".join(PROC_CANDIDATES))


def slot_index(ts):
    return ts.hour * 2 + (1 if ts.minute >= 30 else 0)


def cluster_of_row(row):
    """由 one-hot 還原分群顯示名稱；未命中四群 → 未分群新站。"""
    for c in CLUSTER_ORDER:
        if float(row.get(f"cluster_{c}", 0)) >= 0.5:
            return c, CLUSTER_DISPLAY[c]
    return UNKNOWN_KEY, UNKNOWN_DISPLAY


def geo_tags_of_row(row):
    tags = [t for t in STATION_TYPES if float(row.get(f"type_{t}", 0)) >= 0.5]
    return tags


def to_count(ratio, total):
    """比例 → 車數／空位數；限制在 [0, total]。ratio 非有限則回 None（資料不足）。"""
    if ratio is None or not np.isfinite(ratio):
        return None
    return int(round(max(0.0, min(1.0, float(ratio))) * total))


def pair_counts(bike_ratio, dock_ratio, total):
    """把可借比例、可還比例換算成車數/空位數；相加超過容量則等比例縮放後再取整。"""
    if bike_ratio is None or dock_ratio is None \
            or not np.isfinite(bike_ratio) or not np.isfinite(dock_ratio) or total <= 0:
        return None, None
    b = max(0.0, min(1.0, float(bike_ratio)))
    d = max(0.0, min(1.0, float(dock_ratio)))
    s = b + d
    if s > 1.0:  # 兩者相加超過容量 → 等比例正規化（與評估端一致）
        b, d = b / s, d / s
    bikes = int(round(b * total))
    docks = int(round(d * total))
    # 硬性上限：車數 + 空位數不得超過容量
    if bikes + docks > total:
        docks = max(0, total - bikes)
    return bikes, docks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-date", default="2026-06-29",
                    help="回放日 (YYYY-MM-DD)，需存在於 test.parquet")
    ap.add_argument("--base-time", default="06:30", help="預設基準時點 (HH:MM)")
    args = ap.parse_args()

    cfg = load_cfg()
    run_id = cfg["run_id"]
    proc_path = find_processed()

    with open(EVAL_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)
    thresholds = report["thresholds"]          # {'bike-30':0.115,...} 5 月校正
    test_event = report["test_june"]["event"]  # 6 月事件指標
    test_reg = report["test_june"]["regression"]

    # 只讀回放日：先讀 cur_time 篩列，降低記憶體
    need_proc = ["row_id", "station_key", "cur_time", "總車柱數", "經度", "緯度",
                 "ratio_bike", "ratio_dock", "ratio_unavailable", "is_continuous",
                 "bike_lag_30", "dock_lag_30"]
    need_proc += [f"cluster_{c}" for c in CLUSTER_ORDER] + [f"cluster_{UNKNOWN_KEY}"]
    need_proc += [f"type_{t}" for t in STATION_TYPES]

    proc = pd.read_parquet(proc_path, columns=need_proc)
    preds = pd.read_parquet(PREDS_PATH)

    proc["_dt"] = pd.to_datetime(proc["cur_time"])
    day_mask = proc["_dt"].dt.strftime("%Y-%m-%d") == args.replay_date
    proc = proc[day_mask].copy()
    if proc.empty:
        raise SystemExit(f"回放日 {args.replay_date} 在 test.parquet 沒有資料")

    preds_day = preds[preds["row_id"].isin(set(proc["row_id"]))].copy()
    merged = proc.merge(
        preds_day[["row_id", "pred_bike-30", "pred_dock-30",
                   "pred_bike-60", "pred_dock-60"]],
        on="row_id", how="left")
    merged["_slot"] = merged["_dt"].apply(slot_index)

    # 逐站彙整
    stations = []
    cluster_counts = {CLUSTER_DISPLAY[c]: 0 for c in CLUSTER_ORDER}
    cluster_counts[UNKNOWN_DISPLAY] = 0
    districts = set()
    insufficient_slot_count = 0
    total_slot_count = 0

    for skey, g in merged.groupby("station_key", sort=False):
        g = g.sort_values("_slot")
        first = g.iloc[0]
        parts = str(skey).split("|")
        district = parts[1] if len(parts) >= 2 else "未知區"
        name = parts[2] if len(parts) >= 3 else str(skey)
        total = int(round(float(first["總車柱數"]))) if np.isfinite(first["總車柱數"]) else 0
        cluster_key, cluster_disp = cluster_of_row(first)
        geo_tags = geo_tags_of_row(first)
        lat = float(first["緯度"]) if np.isfinite(first["緯度"]) else None
        lng = float(first["經度"]) if np.isfinite(first["經度"]) else None
        if lat is None or lng is None:
            continue  # 沒座標無法上圖，跳過

        cluster_counts[cluster_disp] = cluster_counts.get(cluster_disp, 0) + 1
        districts.add(district)

        # 48 格 timeline，缺格以 dataStatus 標記
        by_slot = {int(r["_slot"]): r for _, r in g.iterrows()}
        timeline = []
        slot_status = []  # 每格資料狀態，供前端顯示「資料不足」
        for s in range(48):
            total_slot_count += 1
            r = by_slot.get(s)
            if r is None:
                # 該時點缺觀測 → 資料不足（所有數值欄位為 None，不填 0；狀態放在 slot_status）
                timeline.append([None, None, None, None, None, None, None, None, None, None])
                slot_status.append("missing")
                insufficient_slot_count += 1
                continue
            continuous = float(r.get("is_continuous", 0)) >= 0.5
            lag_ok = np.isfinite(r.get("bike_lag_30", np.nan))
            cur_bikes, cur_docks = pair_counts(r["ratio_bike"], r["ratio_dock"], total)
            # 未來預測（比例 → 車數/空位）
            b30 = r.get("pred_bike-30"); d30 = r.get("pred_dock-30")
            b60 = r.get("pred_bike-60"); d60 = r.get("pred_dock-60")
            bikes30, docks30 = pair_counts(b30, d30, total)
            bikes60, docks60 = pair_counts(b60, d60, total)
            # 資料是否足以做可靠預測（歷史特徵不足→標記；仍保留當下觀測值）
            status = "ok" if (continuous and lag_ok) else "insufficient_history"
            if not (continuous and lag_ok):
                insufficient_slot_count += 1
            slot_status.append(status)
            # timeline row 相容 schema:
            #  [0]=現在車數 [1]=+30車數 [2]=+60車數
            #  [3]=+30可借比例 [4]=+30可還比例 [5]=(保留)
            #  [6]=+60可借比例 [7]=+60可還比例
            #  [8]=+30空位數 [9]=+60空位數  (取代舊「建議文字」，前端已改用門檻規則說明)
            timeline.append([
                cur_bikes, bikes30, bikes60,
                None if b30 is None or not np.isfinite(b30) else round(float(b30), 4),
                None if d30 is None or not np.isfinite(d30) else round(float(d30), 4),
                status,
                None if b60 is None or not np.isfinite(b60) else round(float(b60), 4),
                None if d60 is None or not np.isfinite(d60) else round(float(d60), 4),
                docks30, docks60,
            ])

        # 歷史摘要（非模型直接輸出，供型態表格顯示）：平均可借比例、最忙時段、警示時數
        bike_counts = [tl[0] for tl in timeline if tl[0] is not None]
        avg_occ = (round(100.0 * (sum(bike_counts) / len(bike_counts)) / total)
                   if bike_counts and total > 0 else None)
        # 最忙時段：相鄰時點車數變化絕對值最大者的起始時段
        busiest_slot, busiest_delta = None, -1
        for s in range(47):
            a, b = timeline[s][0], timeline[s + 1][0]
            if a is None or b is None:
                continue
            delta = abs(b - a)
            if delta > busiest_delta:
                busiest_delta, busiest_slot = delta, s
        busiest_hour = SLOTS[busiest_slot] if busiest_slot is not None else None
        # 風險時數：當日落入缺車或滿站事件（比例<=0.10）的時段數 × 0.5 小時
        risk_slots = 0
        for tl in timeline:
            if tl[0] is None or total <= 0:
                continue
            bike_ratio = tl[0] / total
            dock_ratio = max(0, total - tl[0]) / total
            if bike_ratio <= EVENT_RATIO or dock_ratio <= EVENT_RATIO:
                risk_slots += 1

        stations.append({
            "id": f"S{len(stations)+1:04d}",
            "key": str(skey),
            "name": name,
            "district": district,
            "pattern": cluster_disp,
            "clusterKey": cluster_key,
            "isUnclustered": cluster_key == UNKNOWN_KEY,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "total": total,
            "geoTags": geo_tags,
            "timeline": timeline,
            "slotStatus": slot_status,
            # 歷史摘要欄位（型態表格用；非模型直接預測）
            "avgOccupancy": (f"{avg_occ}%" if avg_occ is not None else "資料不足"),
            "busiestHour": (busiest_hour or "資料不足"),
            "riskHours": (f"{risk_slots * 0.5:g} 小時" if bike_counts else "資料不足"),
        })

    station_count = len(stations)
    model_cluster_counts = {CLUSTER_DISPLAY[c]: cluster_counts.get(CLUSTER_DISPLAY[c], 0)
                            for c in CLUSTER_ORDER}
    unclustered_count = cluster_counts.get(UNKNOWN_DISPLAY, 0)
    ordered_counts = {**model_cluster_counts}
    if unclustered_count:
        ordered_counts[UNKNOWN_DISPLAY] = unclustered_count

    # 觀測數（回放日實際列數）
    observation_count = int(len(merged))

    meta = {
        "mode": "replay",
        "runId": run_id,
        "replayDate": args.replay_date,
        "baseTime": args.base_time,
        "dataPeriod": "訓練 2026.01–04 · 驗證 2026.05 · 測試 2026.06",
        "dataUpdatedAt": args.replay_date + " " + "（歷史回放，非即時）",
        "slots": SLOTS,
        "stationCount": station_count,
        "districtCount": len(districts),
        "observationCount": observation_count,
        "clusterCounts": ordered_counts,
        "clusterModelCount": len(CLUSTER_ORDER),
        "unclusteredStationCount": unclustered_count,
        "unclusteredLabel": UNKNOWN_DISPLAY,
        # 四向、比例回歸模型的『警示門檻』（非機率）。方向：可借/可還『比例 <= 門檻』→ 缺車/滿站
        "warningThresholds": {
            "bike-30": thresholds["bike-30"],
            "dock-30": thresholds["dock-30"],
            "bike-60": thresholds["bike-60"],
            "dock-60": thresholds["dock-60"],
        },
        "eventDefinition": "真值可借/可還比例 <= 0.10 視為缺車/滿站事件；預測比例 <= 各向門檻即發預警。",
        "modelKind": "XGBoost 比例回歸 (reg:squarederror)，輸出可借/可還比例，非機率。",
        "modelVersion": {
            "runId": run_id,
            "jobs": {t: f"{run_id}-{t}-r2" for t in ["bike-30", "dock-30", "bike-60", "dock-60"]},
            "framework": "SageMaker XGBoost " + cfg["compute"]["xgboost_framework_version"],
        },
        # 6 月測試指標（供稽核，非機率校準）
        "testMetrics": {
            t: {
                "MAE_ratio": round(test_reg[t]["regression"]["MAE"], 4),
                "RMSE_ratio": round(test_reg[t]["regression"]["RMSE"], 4),
                "R2": round(test_reg[t]["regression"]["R2"], 4),
                "event_threshold": test_event[t]["threshold"],
                "event_Precision": round(test_event[t]["Precision"], 4),
                "event_Recall": round(test_event[t]["Recall"], 4),
                "event_F1": round(test_event[t]["F1"], 4),
            } for t in ["bike-30", "dock-30", "bike-60", "dock-60"]
        },
        "insufficientSlotCount": insufficient_slot_count,
        "totalSlotCount": total_slot_count,
        "sourceNote": (f"資料來源：本次 RUN {run_id} 四個 SageMaker XGBoost 訓練模型（-r2）於"
                       f" {args.replay_date} 的可追溯回放預測；車數／空位數由模型輸出比例乘以"
                       f"『總車柱數』換算。尚無推論 Endpoint，未提供即時模式。"),
        "liveMode": {
            "available": False,
            "reason": "SageMaker 尚未部署推論 Endpoint（list-endpoints 為空）。",
            "requirements": [
                "部署四個模型工件為 SageMaker Endpoint（或以一個多模型 Endpoint 承載）",
                "持續更新的站況來源與至少過去 2 小時（30/60/90/120 分鐘）快照以建立 lag 特徵",
                "即時站點代號 ↔ 訓練場站鍵對應、營運狀態與資料新鮮度檢查",
                "後端批次推論並快取整批站點結果，前端只呼叫自家 API",
            ],
        },
    }

    frontend_payload = {"meta": meta, "stations": stations}

    os.makedirs(FRONTEND, exist_ok=True)
    js_path = os.path.join(FRONTEND, "appData.js")
    # 保留朋友的舊靜態資料（只備份一次，不覆寫既有備份）
    legacy_path = os.path.join(FRONTEND, "appData.legacy.js")
    if os.path.exists(js_path) and not os.path.exists(legacy_path):
        import shutil
        shutil.copy2(js_path, legacy_path)
    header = ("// 本檔由 src/build_frontend_data.py 自本次 RUN "
              + run_id + " 四個 SageMaker XGBoost 模型的可追溯回放預測產生。\n"
              "// 車數／空位數 = 模型輸出比例 × 總車柱數。請勿手動編輯。\n")
    js = header + "window.appData = " + json.dumps(frontend_payload, ensure_ascii=False) + ";\n"
    js += "window.allStations = window.appData.stations;\n"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js)

    meta_path = os.path.join(FRONTEND, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 摘要輸出（供終端稽核）
    summary = {
        "replay_date": args.replay_date,
        "station_count": station_count,
        "district_count": len(districts),
        "cluster_counts": ordered_counts,
        "unclustered_count": unclustered_count,
        "observation_count": observation_count,
        "insufficient_slots": insufficient_slot_count,
        "total_slots": total_slot_count,
        "thresholds": meta["warningThresholds"],
        "appData_bytes": os.path.getsize(js_path),
    }
    with open(os.path.join(ROOT, "_build_frontend_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("BUILD_FRONTEND_DONE", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
