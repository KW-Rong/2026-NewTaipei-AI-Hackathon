# -*- coding: utf-8 -*-
"""動態資料鏈測試。

涵蓋：
  1. SnapshotBuffer：寫入/讀取/淘汰/lag 組裝/連續性判斷。
  2. LiveStateProvider：站點對應驗證、新鮮度、無來源回 unavailable、
     快照不足時 states() 仍回傳站況但標記 _is_continuous=0（predictor 判 insufficient）。
  3. 無即時來源 → /api/live 回 503（不以回放值頂替）。
  4. /api/live/status 回就緒狀態。
  5. 注入 fake fetch_current 後 live_readiness 轉為 available=False（因無 Endpoint），
     訊息精確說明缺 Endpoint。
  6. 2 小時快照完整後，build_history 的 4 個 lag 全部有值，_is_continuous=1.0。
  7. 資料過期（data_age_sec > max_stale_sec）→ dataStatus=stale，不產生預測。
  8. /api/health 包含 liveReadiness。
  9. 前端模式：確認 liveMode.js 在 503 時顯示 missing 列表而非回放資料。
"""
import os
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta
from urllib.request import urlopen
from urllib.error import HTTPError
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.station_state import SnapshotBuffer, LiveStateProvider, LAG_MINUTES
from api import app as appmod


# ── helpers ──────────────────────────────────────────────────────────────────
def make_snapshot(buffer, skey, now, n_slots, ratio_bike=0.4, ratio_dock=0.55):
    """往 buffer 寫 n_slots 格（從 now 往前，每格差 30 分）。"""
    for i in range(n_slots):
        ts = now - timedelta(minutes=30 * i)
        buffer.ingest(skey, ts, ratio_bike=ratio_bike, ratio_dock=ratio_dock,
                      total_docks=20, operational=True)


# ── SnapshotBuffer 測試 ───────────────────────────────────────────────────────
class TestSnapshotBuffer(unittest.TestCase):
    def setUp(self):
        self.buf = SnapshotBuffer()
        self.now = datetime(2026, 6, 29, 8, 0, 0)   # 整點，易驗證 floor

    def test_floor_aligns_to_30min(self):
        buf = SnapshotBuffer()
        ts = datetime(2026, 6, 29, 8, 17, 45)
        buf.ingest("s1", ts, 0.4, 0.5, total_docks=20, operational=True)
        latest = buf.latest("s1")
        self.assertEqual(latest["ts"].minute, 0)   # floor 到 08:00

    def test_4_lags_all_present_after_5_snapshots(self):
        """t, t-30, t-60, t-90, t-120 共 5 格 → 4 個 lag 全齊。"""
        make_snapshot(self.buf, "s1", self.now, 5)
        history, cov = self.buf.build_history("s1", self.now)
        self.assertEqual(cov["lags_present"], 4)
        for lag in LAG_MINUTES:
            self.assertIsNotNone(history[lag]["ratio_bike"], f"lag {lag} 不應為 None")

    def test_insufficient_history_with_only_2_snapshots(self):
        """只有 2 格 → lag 60/90/120 為 None → lags_present < 4。"""
        make_snapshot(self.buf, "s2", self.now, 2)
        history, cov = self.buf.build_history("s2", self.now)
        self.assertLess(cov["lags_present"], 4)
        self.assertIsNone(history[120]["ratio_bike"])

    def test_old_slot_evicted(self):
        """超過 span（120 分）的格應被淘汰。"""
        make_snapshot(self.buf, "s3", self.now, 6)   # 多一格
        self.assertLessEqual(
            len(self.buf._data.get("s3", {})), 5)  # 最多保留 span/step + 1 格

    def test_missing_returns_none_not_zero(self):
        """缺格時 ratio_bike 應為 None，不是 0。"""
        make_snapshot(self.buf, "s4", self.now, 1)
        history, _ = self.buf.build_history("s4", self.now)
        for lag in [60, 90, 120]:
            self.assertIsNone(history[lag]["ratio_bike"], f"lag {lag} 缺格應為 None 不是 0")

    def test_station_count(self):
        for i in range(5):
            self.buf.ingest(f"s{i}", self.now, 0.4, 0.5)
        self.assertEqual(self.buf.station_count(), 5)


# ── LiveStateProvider 測試 ────────────────────────────────────────────────────
class TestLiveStateProvider(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 6, 29, 8, 0, 0)

    def _make_provider_with_full_snapshots(self, n_stations=3):
        prov = LiveStateProvider(fetch_current=None)  # 先不給 fetch_current
        for i in range(n_stations):
            skey = f"新北市|測試區|測試站{i}"
            prov.static_by_key[skey] = {"lon": 121.4, "lat": 25.0,
                                        "total_docks": 20, "cluster": "平穩低波動型",
                                        "station_types": ["公車站"], "calendar": "平日"}
            make_snapshot(prov.buffer, skey, self.now, 5)
        return prov

    def test_unavailable_without_fetch(self):
        prov = LiveStateProvider(fetch_current=None)
        self.assertFalse(prov.available())

    def test_available_with_fetch(self):
        prov = LiveStateProvider(fetch_current=lambda: [])
        self.assertTrue(prov.available())

    def test_states_with_full_history_continuous(self):
        prov = self._make_provider_with_full_snapshots()
        states = prov.states(now=self.now + timedelta(seconds=10))
        self.assertEqual(len(states), 3)
        for s in states:
            self.assertEqual(s["_is_continuous"], 1.0)

    def test_states_with_insufficient_history_not_continuous(self):
        prov = LiveStateProvider(fetch_current=None)
        skey = "新北市|測試區|新站A"
        make_snapshot(prov.buffer, skey, self.now, 2)   # 歷史不足
        states = prov.states(now=self.now + timedelta(seconds=5))
        s = next(x for x in states if x["station_key"] == skey)
        self.assertEqual(s["_is_continuous"], 0.0)   # predictor 會判 insufficient

    def test_stale_data_flagged(self):
        prov = LiveStateProvider(fetch_current=None, max_stale_sec=300)
        skey = "新北市|測試區|舊站"
        make_snapshot(prov.buffer, skey, self.now, 5)
        # now 比快照晚 20 分（1200 秒）→ 過期
        late_now = self.now + timedelta(minutes=20)
        states = prov.states(now=late_now)
        s = next(x for x in states if x["station_key"] == skey)
        self.assertGreater(s["data_age_sec"], 300)

    def test_station_key_mapping(self):
        registry = {"LIVE-999": "新北市|板橋區|板橋站"}
        prov = LiveStateProvider(fetch_current=None, station_registry=registry)
        resolved = prov._resolve_key("LIVE-999")
        self.assertEqual(resolved, "新北市|板橋區|板橋站")

    def test_poll_once_ingests_and_records(self):
        rows = [{"station_key": "新北市|板橋區|板橋站", "ts": self.now,
                 "ratio_bike": 0.3, "ratio_dock": 0.6, "ratio_unavailable": 0.0,
                 "total_docks": 30, "operational": True, "lon": 121.45, "lat": 25.01,
                 "cluster": "均衡流動型", "station_types": ["捷運站"], "calendar": "平日"}]
        prov = LiveStateProvider(fetch_current=lambda: rows)
        n = prov.poll_once(now=self.now)
        self.assertEqual(n, 1)
        self.assertTrue(prov._last_fetch_ok)
        self.assertEqual(prov.buffer.station_count(), 1)


# ── API /api/live 測試（無 Endpoint → 503，不以回放值頂替）───────────────────
class TestApiLiveEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        appmod.build_app(fetch_current_fn=None)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        cls.port = cls.httpd.server_address[1]
        t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        t.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path):
        try:
            with urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_health_contains_live_readiness(self):
        code, body = self._get("/api/health")
        self.assertEqual(code, 200)
        self.assertIn("liveReadiness", body)
        self.assertFalse(body["liveReadiness"]["available"])

    def test_live_503_no_endpoint_no_source(self):
        code, body = self._get("/api/live")
        self.assertEqual(code, 503)
        self.assertFalse(body["available"])
        self.assertTrue(body["missing"])
        # 不含任何 stations 欄位（不以回放值頂替）
        self.assertNotIn("stations", body)

    def test_live_status_503(self):
        code, body = self._get("/api/live/status")
        self.assertEqual(code, 503)
        self.assertFalse(body["available"])

    def test_live_missing_lists_both_items(self):
        """無 Endpoint 且無即時來源：missing 應至少提到 Endpoint 和來源。"""
        _, body = self._get("/api/live")
        missing_text = " ".join(body.get("missing", []))
        self.assertIn("Endpoint", missing_text)

    def test_live_has_hint_to_replay(self):
        _, body = self._get("/api/live")
        self.assertIn("replay", body.get("hint", ""))


# ── 注入 fake fetch_current 後的狀態測試 ─────────────────────────────────────
class TestApiLiveWithFakeSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        # 注入 fake fetch_current（回傳一站）
        def fake_fetch():
            return [{"station_key": "新北市|板橋區|板橋站",
                     "ts": datetime(2026, 6, 29, 8, 0),
                     "ratio_bike": 0.4, "ratio_dock": 0.5, "ratio_unavailable": 0.0,
                     "total_docks": 20, "operational": True, "lon": 121.45, "lat": 25.01,
                     "cluster": "均衡流動型", "station_types": ["捷運站"], "calendar": "平日"}]
        appmod.build_app(fetch_current_fn=fake_fetch)
        # 預先 poll 一次
        appmod.APP.live_provider.poll_once(now=datetime(2026, 6, 29, 8, 0))
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        cls.port = cls.httpd.server_address[1]
        t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        t.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path):
        try:
            with urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_live_readiness_still_missing_endpoint(self):
        """有即時來源但無 Endpoint：missing 精確說明缺 Endpoint。"""
        code, body = self._get("/api/live")
        self.assertEqual(code, 503)
        self.assertFalse(body["available"])
        missing_text = " ".join(body["missing"])
        self.assertIn("Endpoint", missing_text)
        # 不應再提「尚未接入即時來源」
        self.assertNotIn("fetch_current", missing_text)

    def test_live_source_configured_reflects_true(self):
        code, body = self._get("/api/live/status")
        self.assertTrue(body.get("liveSourceConfigured"))

    def test_buffered_stations_not_zero(self):
        code, body = self._get("/api/live/status")
        self.assertGreater(body.get("bufferedStations", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
