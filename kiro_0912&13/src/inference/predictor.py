# -*- coding: utf-8 -*-
"""推論協調器：站況批次 → 四個模型比例 → 車數/空位 + 缺車/滿站預警。

職責：
  - 讀 feature_spec.json（本次訓練規格）與 5 月校正門檻（evaluation_report.json）。
  - 對整批站況一次組特徵、一次批次推論（不逐站呼叫模型）。
  - 比例 → 車數/空位：round(比例 × 總車柱數)，並限制 bikes+docks<=容量（配對正規化）。
  - 四向門檻（bike-30/dock-30/bike-60/dock-60）產生預警：預測可借/可還比例 <= 門檻。
  - 站況歷史不足（30/60/90/120 lag 任一缺、或 2h 不連續）→ 該站標 insufficient，不輸出預測。
  - 停用站 / 資料過期 → 標記狀態，不產生預警。

輸出對每站：dataStatus、bikesNow、predictions{horizon:{bikeRatio,dockRatio,bikes,docks}}、
alerts[]、以及可追溯的模型版本與門檻。
"""
import json
import math
import os

from . import features as F

TARGETS = ["bike-30", "dock-30", "bike-60", "dock-60"]
HORIZONS = [30, 60]
EVENT_RATIO = 0.10  # 事件定義（真值比例 <=0.10 視為缺車/滿站），與 5 月校正一致


class Predictor:
    def __init__(self, spec, thresholds, source, stale_after_sec=900):
        self.spec = spec
        self.thresholds = thresholds  # {'bike-30':0.115,...}
        self.source = source          # LocalArtifactPredictor / SageMakerEndpointClient
        self.stale_after_sec = stale_after_sec

    # ---- 換算工具 ----
    @staticmethod
    def _clip01(x):
        if x is None or not math.isfinite(x):
            return None
        return max(0.0, min(1.0, float(x)))

    def _pair_counts(self, bike_ratio, dock_ratio, total):
        b = self._clip01(bike_ratio)
        d = self._clip01(dock_ratio)
        if b is None or d is None or total is None or total <= 0:
            return None, None, b, d
        s = b + d
        nb, nd = (b / s, d / s) if s > 1.0 else (b, d)  # 配對正規化
        bikes = int(round(nb * total))
        docks = int(round(nd * total))
        if bikes + docks > total:  # 硬性容量限制
            docks = max(0, total - bikes)
        return bikes, docks, b, d

    def _threshold_for(self, kind, horizon):
        return self.thresholds[f"{kind}-{horizon}"]

    # ---- 主流程 ----
    def predict_batch(self, states):
        """states: list[StationState dict]。回傳 list[result dict]，順序對齊輸入。"""
        # 1) 組特徵，決定哪些站可預測
        base_list = []
        flags_list = []
        for st in states:
            fd, flags = F.build_base_feature_dict(st, self.spec)
            base_list.append(fd)
            flags_list.append(flags)

        # 2) 只對「可預測」站組向量批次推論
        vectors_by_target = {t: [] for t in TARGETS}
        predictable_index = []  # (station_i) 可預測站索引
        row_map = {t: [] for t in TARGETS}  # 對應到 station_i
        for i, (st, fd, flags) in enumerate(zip(states, base_list, flags_list)):
            can_predict = flags["history_ok"] and flags.get("operational", True)
            # 資料過期判斷
            age = st.get("data_age_sec")
            if age is not None and age > self.stale_after_sec:
                can_predict = False
                flags["reasons"].append(f"資料過期（{int(age)} 秒 > {self.stale_after_sec} 秒）")
                flags["stale"] = True
            if not can_predict:
                continue
            predictable_index.append(i)
            for horizon in HORIZONS:
                for kind in ("bike", "dock"):
                    t = f"{kind}-{horizon}"
                    vectors_by_target[t].append(F.build_vector(fd, self.spec, horizon))
                    row_map[t].append(i)

        # 3) 批次推論
        ratios = self.source.predict_ratios(vectors_by_target) if predictable_index else {t: [] for t in TARGETS}

        # 4) 回填每站預測（比例→車數/空位）與預警
        # 先把 target 級預測攤回 station_i
        pred_by_station = {i: {} for i in predictable_index}
        for t in TARGETS:
            arr = ratios.get(t, [])
            for row_i, station_i in enumerate(row_map[t]):
                if row_i < len(arr):
                    pred_by_station[station_i][t] = arr[row_i]

        results = []
        for i, (st, fd, flags) in enumerate(zip(states, base_list, flags_list)):
            total = fd["總車柱數"]
            total = int(round(total)) if total and math.isfinite(total) else 0
            skey = st.get("station_key")
            cur_bike_ratio = self._clip01(F._num(st.get("ratio_bike")))
            cur_dock_ratio = self._clip01(F._num(st.get("ratio_dock")))
            bikes_now = int(round(cur_bike_ratio * total)) if (cur_bike_ratio is not None and total > 0) else None

            # 地理/識別屬性（非預測值，供前端地圖/列表顯示）：由站況直接帶出
            _parts = str(skey).split("|") if skey else []
            base = {
                "stationKey": skey,
                "name": st.get("name") or (_parts[2] if len(_parts) >= 3 else skey),
                "district": st.get("district") or (_parts[1] if len(_parts) >= 2 else ""),
                "lat": F._num(st.get("lat")),
                "lon": F._num(st.get("lon")),
                "cluster": fd.get("_cluster"),
                "total": total,
                "curTime": st.get("cur_time"),
                "dataAgeSec": st.get("data_age_sec"),
            }

            if i not in predictable_index:
                results.append({**base,
                                "dataStatus": "insufficient" if not flags.get("stale") else "stale",
                                "operational": flags.get("operational", True),
                                "bikesNow": bikes_now,   # 當下觀測仍可顯示（若有）
                                "predictions": {},
                                "alerts": [],
                                "reasons": flags["reasons"]})
                continue

            preds = pred_by_station.get(i, {})
            predictions = {}
            alerts = []
            for horizon in HORIZONS:
                br = preds.get(f"bike-{horizon}")
                dr = preds.get(f"dock-{horizon}")
                bikes, docks, bclip, dclip = self._pair_counts(br, dr, total)
                predictions[str(horizon)] = {
                    "bikeRatio": None if bclip is None else round(bclip, 4),
                    "dockRatio": None if dclip is None else round(dclip, 4),
                    "bikes": bikes,
                    "docks": docks,
                }
                # 四向門檻預警（預測比例 <= 門檻）
                bt = self._threshold_for("bike", horizon)
                dt = self._threshold_for("dock", horizon)
                if bclip is not None and bclip <= bt:
                    alerts.append({"kind": "shortage", "horizon": horizon,
                                   "ratio": round(bclip, 4), "threshold": bt, "immediate": False})
                if dclip is not None and dclip <= dt:
                    alerts.append({"kind": "full", "horizon": horizon,
                                   "ratio": round(dclip, 4), "threshold": dt, "immediate": False})

            # 即時事件（當下比例 <= 事件定義），與未來預警分開
            if cur_bike_ratio is not None and cur_bike_ratio <= EVENT_RATIO:
                alerts.insert(0, {"kind": "shortage", "horizon": 0, "ratio": round(cur_bike_ratio, 4),
                                  "threshold": EVENT_RATIO, "immediate": True})
            elif cur_dock_ratio is not None and cur_dock_ratio <= EVENT_RATIO:
                alerts.insert(0, {"kind": "full", "horizon": 0, "ratio": round(cur_dock_ratio, 4),
                                  "threshold": EVENT_RATIO, "immediate": True})

            results.append({**base,
                            "dataStatus": "ok",
                            "operational": True,
                            "bikesNow": bikes_now,
                            # 當下比例（未取整，供前端 horizon=0 與 KPI 用同一來源，
                            # 避免與 immediate 事件判斷因取整而不一致）
                            "curBikeRatio": None if cur_bike_ratio is None else round(cur_bike_ratio, 4),
                            "curDockRatio": None if cur_dock_ratio is None else round(cur_dock_ratio, 4),
                            "predictions": predictions,
                            "alerts": alerts,
                            "reasons": []})
        return results

    def meta(self):
        return {
            "modelVersion": self.source.version(),
            "warningThresholds": self.thresholds,
            "eventDefinition": "當下真值可借/可還比例 <= 0.10 視為缺車/滿站即時事件；"
                               "未來預警為模型預測可借/可還比例 <= 各向 5 月校正門檻。",
            "modelKind": "XGBoost 比例回歸 (reg:squarederror)，輸出可借/可還比例，非機率。",
            "featureOrder": self.spec["base_feature_order"] + self.spec["target_time_features"],
            "clusters": F.CLUSTERS + [F.UNKNOWN_CLUSTER],
            "lagMinutes": F.LAG_MINUTES,
            "horizons": HORIZONS,
            "staleAfterSec": self.stale_after_sec,
        }


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(eval_report_path):
    with open(eval_report_path, "r", encoding="utf-8") as f:
        return json.load(f)["thresholds"]
