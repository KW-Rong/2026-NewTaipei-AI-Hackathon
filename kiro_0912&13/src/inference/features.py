# -*- coding: utf-8 -*-
"""特徵組裝：把「站況 + 站點靜態資料 + 目標時間」轉成本次四個模型可吃的特徵向量。

嚴格對齊本次訓練的 feature_spec.json：
  - base_feature_order（47 維）+ target_time_features（7 維）= 54 維／模型。
  - 缺值一律以 feature_spec.medians 填補（與訓練/評估一致），但『站況歷史不足』會另外
    以旗標標記，讓上層拒絕產生預測，而非用中位數硬湊出假預測。
  - 分群 one-hot：平穩低波動型 / 均衡流動型 / 雙尖峰高流動型 / 通勤到達型 / 未知或新場站。
    無法對應訓練場站的站點 → 未知或新場站（不硬分四群）。
  - 場站型別 one-hot、日曆 one-hot 同訓練。

站況輸入（StationState）預期欄位（皆為『目前時點 t』的觀測，比例 0~1）：
  station_key, total_docks(總車柱數), lon, lat, cluster(分群顯示或訓練名),
  station_types(list), calendar(假日/國定假日/平日 或 None),
  ratio_bike, ratio_dock, ratio_unavailable,
  history: {30:{'ratio_bike','ratio_dock'}, 60:..., 90:..., 120:...}  # 過去各時點觀測
  cur_time (ISO 字串), operational(bool 站是否營運), data_age_sec(資料新鮮度秒)

lag/delta/2h 視窗特徵由 history 推導；缺任一 lag → history_ok=False。
"""
import math
from datetime import datetime, timedelta

CLUSTERS = ["平穩低波動型", "均衡流動型", "雙尖峰高流動型", "通勤到達型"]
UNKNOWN_CLUSTER = "未知或新場站"
STATION_TYPES = ["捷運站", "公車站", "學校", "公園", "醫院", "商圈", "停車場"]
CALENDAR_CATS = ["假日", "國定假日", "平日"]
LAG_MINUTES = [30, 60, 90, 120]
HORIZONS = [30, 60]

# 未知或新場站別名（前端顯示名 → 訓練用 one-hot 名）
CLUSTER_ALIASES = {
    "未分群新站": UNKNOWN_CLUSTER,
    "未知或新場站": UNKNOWN_CLUSTER,
    "新場站（無訓練分群）": UNKNOWN_CLUSTER,
}


def _num(v, default=None):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def normalize_cluster(name):
    """把分群名稱標準化成訓練用的類別。無法對應→未知或新場站。"""
    if name in CLUSTERS:
        return name
    if name in CLUSTER_ALIASES:
        return CLUSTER_ALIASES[name]
    return UNKNOWN_CLUSTER


def _cyclical(dt):
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    month = dt.month
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
        "is_morning_peak": 1.0 if (7 <= hour < 9) else 0.0,
        "is_evening_peak": 1.0 if (17 <= hour < 19) else 0.0,
    }


def build_base_feature_dict(state, spec):
    """回傳 (feature_dict, flags)。feature_dict 覆蓋 base_feature_order 全部欄位；
    缺值以 spec['medians'] 補。flags 記錄 history_ok / operational / stale 等。"""
    medians = spec["medians"]
    fd = {}
    flags = {"history_ok": True, "reasons": []}

    total = _num(state.get("total_docks"))
    fd["總車柱數"] = total if total is not None else medians["總車柱數"]
    lon = _num(state.get("lon")); lat = _num(state.get("lat"))
    fd["經度"] = lon if lon is not None else medians["經度"]
    fd["緯度"] = lat if lat is not None else medians["緯度"]

    rb = _num(state.get("ratio_bike")); rd = _num(state.get("ratio_dock"))
    ru = _num(state.get("ratio_unavailable"), 0.0)
    if rb is None or rd is None:
        flags["history_ok"] = False
        flags["reasons"].append("目前站況比例缺失")
    fd["ratio_bike"] = rb if rb is not None else medians["ratio_bike"]
    fd["ratio_dock"] = rd if rd is not None else medians["ratio_dock"]
    fd["ratio_unavailable"] = ru if ru is not None else medians["ratio_unavailable"]

    # lag：過去各時點的比例
    hist = state.get("history", {}) or {}
    lag_vals = {}
    for lag in LAG_MINUTES:
        h = hist.get(lag) or hist.get(str(lag)) or {}
        lb = _num(h.get("ratio_bike")); ld = _num(h.get("ratio_dock"))
        if lb is None or ld is None:
            flags["history_ok"] = False
            flags["reasons"].append(f"缺過去 {lag} 分鐘觀測")
        fd[f"bike_lag_{lag}"] = lb if lb is not None else medians[f"bike_lag_{lag}"]
        fd[f"dock_lag_{lag}"] = ld if ld is not None else medians[f"dock_lag_{lag}"]
        lag_vals[lag] = (lb, ld)

    # delta（現在 - lag）
    for h in [30, 60]:
        lb, ld = lag_vals.get(h, (None, None))
        fd[f"bike_delta_{h}"] = (fd["ratio_bike"] - lb) if (rb is not None and lb is not None) else medians[f"bike_delta_{h}"]
        fd[f"dock_delta_{h}"] = (fd["ratio_dock"] - ld) if (rd is not None and ld is not None) else medians[f"dock_delta_{h}"]

    # 過去兩小時視窗（t, t-30, t-60, t-90）平均/標準差 + 連續性
    win_bike = [rb, lag_vals.get(30, (None,))[0], lag_vals.get(60, (None,))[0], lag_vals.get(90, (None,))[0]]
    win_dock = [rd, lag_vals.get(30, (None, None))[1], lag_vals.get(60, (None, None))[1], lag_vals.get(90, (None, None))[1]]
    wb = [x for x in win_bike if x is not None]
    wd = [x for x in win_dock if x is not None]
    continuous = (len(wb) == 4 and len(wd) == 4)
    if not continuous:
        flags["history_ok"] = False
        flags["reasons"].append("過去 2 小時（30/60/90 分）觀測不連續")
    fd["bike_2h_mean"] = (sum(wb) / len(wb)) if wb else medians["bike_2h_mean"]
    fd["dock_2h_mean"] = (sum(wd) / len(wd)) if wd else medians["dock_2h_mean"]
    fd["bike_2h_std"] = _std(wb) if len(wb) > 1 else medians["bike_2h_std"]
    fd["dock_2h_std"] = _std(wd) if len(wd) > 1 else medians["dock_2h_std"]
    fd["is_continuous"] = 1.0 if continuous else 0.0

    # 目前時間循環特徵
    cur_dt = _parse_dt(state.get("cur_time"))
    if cur_dt is None:
        flags["history_ok"] = False
        flags["reasons"].append("cur_time 無法解析")
        cur_dt = datetime(2026, 6, 1, 0, 0)
    cyc = _cyclical(cur_dt)
    fd["cur_hour_sin"] = cyc["hour_sin"]; fd["cur_hour_cos"] = cyc["hour_cos"]
    fd["cur_dow_sin"] = cyc["dow_sin"]; fd["cur_dow_cos"] = cyc["dow_cos"]
    fd["cur_month_sin"] = cyc["month_sin"]; fd["cur_month_cos"] = cyc["month_cos"]
    fd["cur_is_morning_peak"] = cyc["is_morning_peak"]; fd["cur_is_evening_peak"] = cyc["is_evening_peak"]

    # 日曆 one-hot
    cal = state.get("calendar")
    for c in CALENDAR_CATS:
        fd[f"cal_{c}"] = 1.0 if cal == c else 0.0
    fd["cal_未知"] = 1.0 if cal not in CALENDAR_CATS else 0.0

    # 場站型別 one-hot
    types = set(state.get("station_types") or [])
    for t in STATION_TYPES:
        fd[f"type_{t}"] = 1.0 if t in types else 0.0

    # 分群 one-hot
    cluster = normalize_cluster(state.get("cluster"))
    for c in CLUSTERS:
        fd[f"cluster_{c}"] = 1.0 if cluster == c else 0.0
    fd[f"cluster_{UNKNOWN_CLUSTER}"] = 1.0 if cluster == UNKNOWN_CLUSTER else 0.0

    # 站點狀態旗標
    if state.get("operational") is False:
        flags["reasons"].append("站點停用")
        flags["operational"] = False
    else:
        flags["operational"] = True
    fd["_cur_dt"] = cur_dt  # 內部用，組 target-time 時取用（不進特徵向量）
    fd["_cluster"] = cluster
    return fd, flags


def _std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _parse_dt(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except (ValueError, TypeError):
        return None


def build_vector(base_fd, spec, horizon):
    """依 feature_spec 的欄位順序組出單一模型（該 horizon）的 54 維特徵 list。"""
    base_order = spec["base_feature_order"]
    tgt_feats = spec["target_time_features"]  # 含 horizon_minutes
    cur_dt = base_fd["_cur_dt"]
    tgt_dt = cur_dt + timedelta(minutes=horizon)
    cyc = _cyclical(tgt_dt)
    tgt_map = {
        "tgt_hour_sin": cyc["hour_sin"], "tgt_hour_cos": cyc["hour_cos"],
        "tgt_dow_sin": cyc["dow_sin"], "tgt_dow_cos": cyc["dow_cos"],
        "tgt_is_morning_peak": cyc["is_morning_peak"],
        "tgt_is_evening_peak": cyc["is_evening_peak"],
        "horizon_minutes": float(horizon),
    }
    vec = [float(base_fd[c]) for c in base_order]
    for c in tgt_feats:
        vec.append(float(tgt_map[c]))
    return vec
