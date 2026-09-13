# -*- coding: utf-8 -*-
"""站況來源：把『某時點的整批站況』組成 predictor 需要的 StationState 清單。

即時模式的站況本應來自持續更新的即時來源（例如官方 GBFS/開放資料 + 過去 2 小時快照）。
目前尚未接入即時來源；本模組提供：

  ReplayStateProvider
    - 從本地 processed test.parquet 取某時點各站『當下觀測比例 + 過去 30/60/90/120 分 lag』，
      組成完整 StationState，交給 predictor 重新推論。
    - 這讓『即時推論的資料流』可以用 6 月的真實站況端到端測試（資料流真、模型真、
      只是站況時間是歷史的）。它與『回放預測結果(appData.js)』不同：這裡是後端『重新推論』。

  即時來源（LiveStateProvider）為介面預留：未接入時明確回報 data unavailable，
  不以回放值冒充即時。
"""
import os

import pandas as pd

LAG_MINUTES = [30, 60, 90, 120]
CLUSTER_COLS = {
    "平穩低波動型": "cluster_平穩低波動型",
    "均衡流動型": "cluster_均衡流動型",
    "雙尖峰高流動型": "cluster_雙尖峰高流動型",
    "通勤到達型": "cluster_通勤到達型",
    "未知或新場站": "cluster_未知或新場站",
}
TYPE_COLS = ["捷運站", "公車站", "學校", "公園", "醫院", "商圈", "停車場"]
CAL_COLS = {"假日": "cal_假日", "國定假日": "cal_國定假日", "平日": "cal_平日"}


class ReplayStateProvider:
    """用本地 processed test.parquet 提供某時點整批站況（含 lag），供後端重新推論。"""

    def __init__(self, processed_test_path):
        if not os.path.exists(processed_test_path):
            raise FileNotFoundError(f"找不到 processed test.parquet：{processed_test_path}")
        self.df = pd.read_parquet(processed_test_path)
        self.df["_dt"] = pd.to_datetime(self.df["cur_time"])
        self._dates = sorted(self.df["_dt"].dt.strftime("%Y-%m-%d").unique().tolist())

    def available_dates(self):
        return self._dates

    def _cluster_of(self, row):
        for name, col in CLUSTER_COLS.items():
            if col in row and float(row[col]) >= 0.5:
                return name
        return "未知或新場站"

    def _types_of(self, row):
        return [t for t in TYPE_COLS if f"type_{t}" in row and float(row[f"type_{t}"]) >= 0.5]

    def _calendar_of(self, row):
        for name, col in CAL_COLS.items():
            if col in row and float(row[col]) >= 0.5:
                return name
        return None

    def states_at(self, date_str, time_str):
        """回傳某日某時點（HH:MM）整批站況 StationState list。

        lag 由 processed 已算好的 bike_lag_* / dock_lag_* 欄位直接取用（與訓練一致），
        因此 history 直接帶入這些 lag 值；is_continuous 也一併帶出讓 predictor 判斷。
        """
        ts = pd.Timestamp(f"{date_str} {time_str}:00")
        sub = self.df[self.df["_dt"] == ts]
        states = []
        for _, r in sub.iterrows():
            parts = str(r["station_key"]).split("|")
            district = parts[1] if len(parts) >= 2 else ""
            name = parts[2] if len(parts) >= 3 else str(r["station_key"])
            # history 用 processed 內已算好的 lag 值（等同過去各時點觀測比例）
            history = {}
            for lag in LAG_MINUTES:
                lb = r.get(f"bike_lag_{lag}")
                ld = r.get(f"dock_lag_{lag}")
                history[lag] = {"ratio_bike": None if pd.isna(lb) else float(lb),
                                "ratio_dock": None if pd.isna(ld) else float(ld)}
            states.append({
                "station_key": str(r["station_key"]),
                "district": district,
                "name": name,
                "total_docks": float(r["總車柱數"]),
                "lon": float(r["經度"]),
                "lat": float(r["緯度"]),
                "cluster": self._cluster_of(r),
                "station_types": self._types_of(r),
                "calendar": self._calendar_of(r),
                "ratio_bike": float(r["ratio_bike"]),
                "ratio_dock": float(r["ratio_dock"]),
                "ratio_unavailable": float(r.get("ratio_unavailable", 0.0)),
                "history": history,
                # 直接帶 is_continuous，讓 predictor 判斷歷史是否足夠
                "_is_continuous": float(r.get("is_continuous", 0.0)),
                "cur_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "operational": True,
                "data_age_sec": 0,  # 回放：視為新鮮（歷史重推，非即時）
            })
        return states


# =====================================================================================
# 即時站況：2 小時快照緩衝 + 即時來源抽象（不接假資料；無真來源時回 unavailable）
# =====================================================================================
import threading
import time as _time
from datetime import datetime, timedelta

SNAPSHOT_STEP_MIN = 30           # 訓練用的時間粒度：每 30 分鐘一格
SNAPSHOT_SPAN_MIN = 120          # 需保留過去 2 小時
MAX_STALE_SEC_DEFAULT = 900      # 資料新鮮度上限（15 分鐘）


class SnapshotBuffer:
    """每站過去 2 小時、每 30 分鐘一格的觀測環形緩衝。

    一筆觀測：{'ts': datetime(對齊到 30 分), 'ratio_bike','ratio_dock','ratio_unavailable',
              'total_docks','operational'}
    以 (station_key -> {slot_ts -> obs}) 保存；build_history() 產生 30/60/90/120 分 lag。
    """

    def __init__(self, span_min=SNAPSHOT_SPAN_MIN, step_min=SNAPSHOT_STEP_MIN):
        self.span_min = span_min
        self.step_min = step_min
        self._data = {}  # station_key -> dict[floor_ts_iso -> obs]
        self._lock = threading.Lock()

    @staticmethod
    def _floor(ts, step_min):
        minute = (ts.minute // step_min) * step_min
        return ts.replace(minute=minute, second=0, microsecond=0)

    def ingest(self, station_key, ts, ratio_bike, ratio_dock, ratio_unavailable=0.0,
               total_docks=None, operational=True):
        """寫入一筆對齊到 30 分格的觀測，並淘汰超過 span 的舊格。"""
        floor_ts = self._floor(ts, self.step_min)
        key = floor_ts.strftime("%Y-%m-%d %H:%M")
        with self._lock:
            bucket = self._data.setdefault(station_key, {})
            bucket[key] = {
                "ts": floor_ts, "ratio_bike": ratio_bike, "ratio_dock": ratio_dock,
                "ratio_unavailable": ratio_unavailable, "total_docks": total_docks,
                "operational": operational,
            }
            # 淘汰超過 span 的舊格（保留 span/step 格，不多一格）
            cutoff = floor_ts - timedelta(minutes=self.span_min)
            for k in list(bucket.keys()):
                if bucket[k]["ts"] < cutoff:
                    del bucket[k]

    def build_history(self, station_key, now_ts):
        """回傳 (history_dict, coverage)。history: {30:{ratio_bike,ratio_dock}, ...}。
        缺格則該 lag 為 None（predictor 會判定歷史不足，不臆造）。"""
        floor_now = self._floor(now_ts, self.step_min)
        bucket = self._data.get(station_key, {})
        history = {}
        present = 0
        for lag in LAG_MINUTES:
            slot = (floor_now - timedelta(minutes=lag)).strftime("%Y-%m-%d %H:%M")
            obs = bucket.get(slot)
            if obs is None:
                history[lag] = {"ratio_bike": None, "ratio_dock": None}
            else:
                history[lag] = {"ratio_bike": obs["ratio_bike"], "ratio_dock": obs["ratio_dock"]}
                present += 1
        return history, {"lags_present": present, "lags_needed": len(LAG_MINUTES)}

    def latest(self, station_key):
        bucket = self._data.get(station_key, {})
        if not bucket:
            return None
        return max(bucket.values(), key=lambda o: o["ts"])

    def station_count(self):
        return len(self._data)


class LiveStateProvider:
    """即時站況來源抽象。

    需要注入：
      - fetch_current:  callable() -> list[dict]，每筆為某站『目前』觀測，欄位至少：
          {station_key, ts(datetime或ISO), ratio_bike, ratio_dock, ratio_unavailable,
           total_docks, operational, lon, lat, cluster, station_types, calendar}
        這是真實即時來源的接點（例如官方 GBFS/開放資料 adapter）。未提供時 provider 不可用。
      - station_registry: dict[live_station_id 或 station_key -> 訓練場站鍵]（站點對應驗證用）
        未提供時以 station_key 原樣對應，並在 states() 標記 mapping_verified=False。

    provider 會把每次 fetch 的觀測寫入 SnapshotBuffer，累積足夠 2 小時快照後才可能產生預測。
    """

    def __init__(self, fetch_current=None, station_registry=None,
                 static_by_key=None, max_stale_sec=MAX_STALE_SEC_DEFAULT):
        self.fetch_current = fetch_current
        self.station_registry = station_registry or {}
        # station_key -> 靜態資料（lon/lat/cluster/station_types/calendar/total_docks）
        self.static_by_key = static_by_key or {}
        self.max_stale_sec = max_stale_sec
        self.buffer = SnapshotBuffer()
        self._last_fetch_ok = None
        self._last_error = None

    def available(self):
        return callable(self.fetch_current)

    def status(self):
        return {
            "available": self.available(),
            "bufferedStations": self.buffer.station_count(),
            "lastFetchOk": self._last_fetch_ok,
            "lastError": self._last_error,
            "maxStaleSec": self.max_stale_sec,
            "registrySize": len(self.station_registry),
        }

    @staticmethod
    def _to_dt(ts):
        if isinstance(ts, datetime):
            return ts
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(str(ts), fmt)
            except (ValueError, TypeError):
                continue
        return None

    def poll_once(self, now=None):
        """呼叫一次即時來源，寫入快照緩衝。回傳 ingested 筆數。無來源→拋錯。"""
        if not self.available():
            raise RuntimeError("即時來源未接入（fetch_current 未提供）")
        now = now or datetime.now()
        try:
            rows = self.fetch_current() or []
        except Exception as e:
            self._last_fetch_ok = False
            self._last_error = f"{type(e).__name__}: {e}"
            raise
        n = 0
        for r in rows:
            skey = self._resolve_key(r.get("station_key") or r.get("station_id"))
            ts = self._to_dt(r.get("ts")) or now
            self.buffer.ingest(
                skey, ts,
                ratio_bike=r.get("ratio_bike"), ratio_dock=r.get("ratio_dock"),
                ratio_unavailable=r.get("ratio_unavailable", 0.0),
                total_docks=r.get("total_docks"), operational=r.get("operational", True))
            # 記錄靜態資料（首見即存）
            if skey not in self.static_by_key:
                self.static_by_key[skey] = {
                    "lon": r.get("lon"), "lat": r.get("lat"), "cluster": r.get("cluster"),
                    "station_types": r.get("station_types") or [], "calendar": r.get("calendar"),
                    "total_docks": r.get("total_docks"),
                }
            n += 1
        self._last_fetch_ok = True
        self._last_error = None
        return n

    def _resolve_key(self, raw):
        if raw in self.station_registry:
            return self.station_registry[raw]
        return raw

    def states(self, now=None):
        """由目前緩衝與最新觀測組出整批 StationState（含 2 小時 lag、新鮮度、對應驗證）。

        每站附：
          data_age_sec（新鮮度）、operational、_is_continuous（是否 4 個 lag 齊全）、
          mapping_verified（站點是否在對應表內）。
        資料不足時交給 predictor 判 insufficient，不臆造、不用回放值頂替。
        """
        now = now or datetime.now()
        out = []
        for skey in list(self.buffer._data.keys()):
            latest = self.buffer.latest(skey)
            if latest is None:
                continue
            age = (now - latest["ts"]).total_seconds()
            history, cov = self.buffer.build_history(skey, latest["ts"])
            static = self.static_by_key.get(skey, {})
            parts = str(skey).split("|")
            district = parts[1] if len(parts) >= 2 else ""
            name = parts[2] if len(parts) >= 3 else str(skey)
            out.append({
                "station_key": skey,
                "district": district,
                "name": name,
                "total_docks": static.get("total_docks") or latest.get("total_docks"),
                "lon": static.get("lon"), "lat": static.get("lat"),
                "cluster": static.get("cluster") or "未知或新場站",
                "station_types": static.get("station_types") or [],
                "calendar": static.get("calendar"),
                "ratio_bike": latest["ratio_bike"], "ratio_dock": latest["ratio_dock"],
                "ratio_unavailable": latest.get("ratio_unavailable", 0.0),
                "history": history,
                "_is_continuous": 1.0 if cov["lags_present"] == cov["lags_needed"] else 0.0,
                "cur_time": latest["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                "operational": latest.get("operational", True),
                "data_age_sec": age,
                "mapping_verified": (skey in self.station_registry.values()
                                     or skey in self.station_registry),
                "lagCoverage": cov,
            })
        return out
