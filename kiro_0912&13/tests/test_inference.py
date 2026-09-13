# -*- coding: utf-8 -*-
"""後端推論核心測試（不需 Endpoint；用本地模型工件 + 測試替身）。

涵蓋：
  1. 特徵向量維度與欄位順序對齊 feature_spec（54 維）。
  2. 四種推論（bike-30/dock-30/bike-60/dock-60）都有輸出，且為 0~1 比例。
  3. 比例轉車數：含容量限制（bikes+docks<=total）與配對正規化。
  4. 歷史不足（缺 lag / 2h 不連續）→ dataStatus=insufficient，不輸出預測。
  5. 停用站 / 資料過期 → 不產生預測。
  6. 未知或新場站分群不硬分四群。
  7. 四向門檻預警方向正確（預測比例<=門檻）。
  8. 用『測試替身』(FakePredictorSource) 驗證 predict_batch 流程，不依賴真 Endpoint。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import features as F
from inference.predictor import Predictor, load_spec, load_thresholds

SPEC = load_spec(os.path.join(ROOT, "artifacts", "eval_work", "processed", "feature_spec.json"))
TH = load_thresholds(os.path.join(ROOT, "artifacts", "evaluation_report.json"))


class FakeSource:
    """測試替身：回傳可控比例，驗證換算/門檻/流程，不需模型或 Endpoint。"""

    def __init__(self, ratio_map):
        # ratio_map: {target: fixed_ratio}
        self.ratio_map = ratio_map

    def version(self):
        return {"kind": "fake", "runId": "test"}

    def predict_ratios(self, vectors_by_target):
        return {t: [self.ratio_map[t]] * len(v) for t, v in vectors_by_target.items()}


def full_state(**over):
    """歷史充足的正常站況。"""
    hist = {lag: {"ratio_bike": 0.4, "ratio_dock": 0.55} for lag in [30, 60, 90, 120]}
    st = {
        "station_key": "新北市|測試區|測試站",
        "total_docks": 20, "lon": 121.46, "lat": 25.02,
        "cluster": "平穩低波動型", "station_types": ["公園"], "calendar": "平日",
        "ratio_bike": 0.4, "ratio_dock": 0.55, "ratio_unavailable": 0.0,
        "history": hist, "cur_time": "2026-06-29 06:30:00",
        "operational": True, "data_age_sec": 0,
    }
    st.update(over)
    return st


class TestFeatures(unittest.TestCase):
    def test_vector_dimension_and_order(self):
        fd, flags = F.build_base_feature_dict(full_state(), SPEC)
        self.assertTrue(flags["history_ok"])
        vec30 = F.build_vector(fd, SPEC, 30)
        vec60 = F.build_vector(fd, SPEC, 60)
        self.assertEqual(len(vec30), SPEC["n_features_per_model"])  # 54
        self.assertEqual(len(vec60), SPEC["n_features_per_model"])
        # horizon_minutes 是最後一維
        self.assertEqual(vec30[-1], 30.0)
        self.assertEqual(vec60[-1], 60.0)

    def test_unknown_cluster_not_forced(self):
        fd, _ = F.build_base_feature_dict(full_state(cluster="某個沒見過的分群"), SPEC)
        self.assertEqual(fd["cluster_未知或新場站"], 1.0)
        for c in F.CLUSTERS:
            self.assertEqual(fd[f"cluster_{c}"], 0.0)

    def test_alias_maps_to_unknown(self):
        fd, _ = F.build_base_feature_dict(full_state(cluster="未分群新站"), SPEC)
        self.assertEqual(fd["cluster_未知或新場站"], 1.0)

    def test_history_insufficient_when_lag_missing(self):
        hist = {30: {"ratio_bike": 0.4, "ratio_dock": 0.55},
                60: {"ratio_bike": 0.4, "ratio_dock": 0.55},
                90: {"ratio_bike": None, "ratio_dock": None},  # 缺 90 分
                120: {"ratio_bike": 0.4, "ratio_dock": 0.55}}
        fd, flags = F.build_base_feature_dict(full_state(history=hist), SPEC)
        self.assertFalse(flags["history_ok"])


class TestPredictorPairing(unittest.TestCase):
    def test_ratio_to_counts_capacity_limit(self):
        # bike=0.7, dock=0.7 → 相加 1.4 >1 → 正規化各 0.5 → total 20 → 10/10
        src = FakeSource({"bike-30": 0.7, "dock-30": 0.7, "bike-60": 0.7, "dock-60": 0.7})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state()])[0]
        self.assertEqual(res["dataStatus"], "ok")
        p30 = res["predictions"]["30"]
        self.assertLessEqual(p30["bikes"] + p30["docks"], res["total"])
        self.assertEqual(p30["bikes"], 10)
        self.assertEqual(p30["docks"], 10)

    def test_four_predictions_present_and_ratio_range(self):
        src = FakeSource({"bike-30": 0.3, "dock-30": 0.6, "bike-60": 0.25, "dock-60": 0.65})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state()])[0]
        for h in ("30", "60"):
            self.assertIn(h, res["predictions"])
            for k in ("bikeRatio", "dockRatio"):
                v = res["predictions"][h][k]
                self.assertTrue(0.0 <= v <= 1.0)

    def test_shortage_alert_direction(self):
        # bike-30 比例 0.05 <= 0.115 → 缺車預警
        src = FakeSource({"bike-30": 0.05, "dock-30": 0.9, "bike-60": 0.5, "dock-60": 0.5})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state()])[0]
        kinds = [(a["kind"], a["horizon"]) for a in res["alerts"]]
        self.assertIn(("shortage", 30), kinds)
        for a in res["alerts"]:
            if a["kind"] == "shortage":
                self.assertLessEqual(a["ratio"], a["threshold"])

    def test_immediate_event_separated(self):
        # 當下可借比例 0.05 <=0.10 → 即時事件 immediate=True，horizon=0
        src = FakeSource({"bike-30": 0.5, "dock-30": 0.5, "bike-60": 0.5, "dock-60": 0.5})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state(ratio_bike=0.05, ratio_dock=0.9)])[0]
        self.assertTrue(any(a["immediate"] and a["horizon"] == 0 for a in res["alerts"]))


class TestPredictorDataStatus(unittest.TestCase):
    def test_insufficient_history_no_prediction(self):
        hist = {30: {"ratio_bike": 0.4, "ratio_dock": 0.55}}  # 只有 30 分
        src = FakeSource({"bike-30": 0.3, "dock-30": 0.6, "bike-60": 0.3, "dock-60": 0.6})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state(history=hist)])[0]
        self.assertEqual(res["dataStatus"], "insufficient")
        self.assertEqual(res["predictions"], {})
        self.assertEqual(res["alerts"], [])
        self.assertTrue(res["reasons"])

    def test_stale_data_no_prediction(self):
        src = FakeSource({"bike-30": 0.3, "dock-30": 0.6, "bike-60": 0.3, "dock-60": 0.6})
        pred = Predictor(SPEC, TH, src, stale_after_sec=900)
        res = pred.predict_batch([full_state(data_age_sec=5000)])[0]
        self.assertEqual(res["dataStatus"], "stale")
        self.assertEqual(res["predictions"], {})

    def test_non_operational_no_prediction(self):
        src = FakeSource({"bike-30": 0.3, "dock-30": 0.6, "bike-60": 0.3, "dock-60": 0.6})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state(operational=False)])[0]
        self.assertEqual(res["predictions"], {})
        self.assertFalse(res["operational"])

    def test_missing_not_zero(self):
        # 資料不足時，bikes 不應被硬填 0（predictions 為空，而非 0）
        hist = {30: {"ratio_bike": 0.4, "ratio_dock": 0.55}}
        src = FakeSource({"bike-30": 0.3, "dock-30": 0.6, "bike-60": 0.3, "dock-60": 0.6})
        pred = Predictor(SPEC, TH, src)
        res = pred.predict_batch([full_state(history=hist)])[0]
        self.assertNotIn("30", res["predictions"])


class TestLocalArtifactModel(unittest.TestCase):
    """實際載入本次四個模型工件（本地 model.json）做一次真推論。"""

    @classmethod
    def setUpClass(cls):
        from inference.sources import LocalArtifactPredictor
        cls.src = LocalArtifactPredictor(
            os.path.join(ROOT, "artifacts", "eval_work"), "youbike-20260912-122927")

    def test_local_artifact_four_models_predict(self):
        pred = Predictor(SPEC, TH, self.src)
        res = pred.predict_batch([full_state(), full_state(cluster="通勤到達型")])
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertEqual(r["dataStatus"], "ok")
            self.assertIn("30", r["predictions"])
            self.assertIn("60", r["predictions"])
            for h in ("30", "60"):
                self.assertTrue(0.0 <= r["predictions"][h]["bikeRatio"] <= 1.0)
                self.assertTrue(0.0 <= r["predictions"][h]["dockRatio"] <= 1.0)
                self.assertLessEqual(
                    r["predictions"][h]["bikes"] + r["predictions"][h]["docks"], r["total"])

    def test_version_marks_local_not_endpoint(self):
        v = self.src.version()
        self.assertEqual(v["kind"], "local_artifact")
        self.assertIn("非線上", v["note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
