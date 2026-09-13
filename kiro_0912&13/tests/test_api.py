# -*- coding: utf-8 -*-
"""API 契約測試：直接建立 AppState 並呼叫其方法（不需真的開 port），
外加一個以 http.server 實際起服務、用 urllib 打 /api/* 的端到端本機測試。

不需要也不會呼叫真實 SageMaker Endpoint：mode 會是 local_artifact。
"""
import json
import os
import sys
import threading
import time
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from api import app as appmod


class TestAppStateReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = appmod.build_app()

    def test_mode_is_local_artifact_no_endpoint(self):
        # 未設 YOUBIKE_SM_ENDPOINTS → 應為本地工件推論，且 hasEndpoint=False
        self.assertEqual(self.app.mode, "local_artifact")
        self.assertFalse(self.app.has_endpoint)

    def test_meta_contains_thresholds_and_version(self):
        m = self.app.predictor.meta()
        self.assertEqual(m["warningThresholds"]["bike-30"], 0.115)
        self.assertEqual(m["warningThresholds"]["dock-60"], 0.105)
        self.assertEqual(m["modelVersion"]["kind"], "local_artifact")
        self.assertIn("平穩低波動型", m["clusters"])
        self.assertIn("未知或新場站", m["clusters"])

    def test_replay_returns_traceable_stations(self):
        payload = self.app.replay("2026-06-29", "06:30")
        self.assertEqual(payload["mode"], "replay")
        self.assertTrue(payload["traceable"])
        self.assertEqual(payload["dataTime"], "2026-06-29 06:30")
        self.assertGreater(payload["stationCount"], 1000)
        # 每站至少有 dataStatus；ok 站有四種預測比例
        ok = [s for s in payload["stations"] if s["dataStatus"] == "ok"]
        self.assertTrue(ok)
        s = ok[0]
        for h in ("30", "60"):
            self.assertIn(h, s["predictions"])
            self.assertTrue(0.0 <= s["predictions"][h]["bikeRatio"] <= 1.0)
            self.assertTrue(0.0 <= s["predictions"][h]["dockRatio"] <= 1.0)
            self.assertLessEqual(
                s["predictions"][h]["bikes"] + s["predictions"][h]["docks"], s["total"])

    def test_replay_cached(self):
        t0 = time.time()
        self.app.replay("2026-06-29", "07:00")
        first = time.time() - t0
        t1 = time.time()
        self.app.replay("2026-06-29", "07:00")  # 第二次應命中快取
        second = time.time() - t1
        self.assertLessEqual(second, first + 0.01)

    def test_live_not_available_without_endpoint(self):
        status = self.app.live_status()
        self.assertFalse(status["liveAvailable"])
        self.assertFalse(status["hasEndpoint"])
        self.assertTrue(status["missing"])

    def test_insufficient_or_stale_marked_not_zero(self):
        payload = self.app.replay("2026-06-29", "06:30")
        # 找資料不足站（若有）：predictions 應為空，而非填 0
        insuff = [s for s in payload["stations"] if s["dataStatus"] != "ok"]
        for s in insuff:
            self.assertEqual(s["predictions"], {})
            self.assertEqual(s["alerts"], [])


class TestHttpServerEndToEnd(unittest.TestCase):
    """實際起 http.server（本機 127.0.0.1 隨機埠），用 urllib 打端點。"""

    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer
        appmod.build_app()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path):
        with urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def test_health(self):
        code, body = self._get("/api/health")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["hasEndpoint"])

    def test_meta(self):
        code, body = self._get("/api/meta")
        self.assertEqual(code, 200)
        self.assertIn("warningThresholds", body)
        self.assertIn("replayDates", body)

    def test_replay_endpoint(self):
        code, body = self._get("/api/replay?date=2026-06-29&time=06:30")
        self.assertEqual(code, 200)
        self.assertEqual(body["mode"], "replay")
        self.assertGreater(body["stationCount"], 1000)

    def test_live_endpoint_503(self):
        try:
            code, body = self._get("/api/live")
        except HTTPError as e:
            code = e.code
            body = json.loads(e.read().decode("utf-8"))
        self.assertEqual(code, 503)
        self.assertFalse(body["available"])
        self.assertTrue(body["missing"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
