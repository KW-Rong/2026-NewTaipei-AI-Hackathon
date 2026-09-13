# -*- coding: utf-8 -*-
"""YouBike 後端 API（Python 內建 http.server，零額外依賴）。

前端只呼叫這些端點；瀏覽器不放 AWS 憑證、也不直接呼叫 SageMaker。
後端一次批次推論整批站點並快取（不逐站呼叫模型）。

端點：
  GET /api/health
      → {status, mode, hasEndpoint, liveStatus, ...}
  GET /api/meta
      → 模型版本、四向門檻、事件定義、分群、特徵順序、liveStatus 等
  GET /api/replay?date=YYYY-MM-DD&time=HH:MM
      → 歷史回放（可追溯本次四個模型工件）；後端『重新推論』該時點整批站況
      → {mode:'replay', dataTime, source, modelVersion, stations:[...], ...}
  GET /api/live
      → 即時預測：
          若尚未就緒（無 Endpoint / 無即時站況 / 快照不足）→ 503，列缺什麼
          就緒時 → 後端批次呼叫 Endpoint + 套門檻，回 {mode:'live',...,stations:[...]}
          不以回放值冒充即時。
  GET /api/live/status
      → 即時模式就緒狀態明細（用於前端即時顯示狀態，不含 stations）

環境變數（皆選用；不含任何憑證於程式）：
  YOUBIKE_SM_ENDPOINTS   JSON 字串，設定則用線上 Endpoint，否則用本地工件推論
  YOUBIKE_LIVE_SOURCE    非空字串=即時站況來源就緒（需同時注入 fetch_current_fn）
  YOUBIKE_API_PORT       預設 8000
  YOUBIKE_AWS_PROFILE    伺服器端 boto3 profile（選用）
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
import sys
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference.predictor import Predictor, load_spec, load_thresholds  # noqa: E402
from inference.sources import make_predictor                           # noqa: E402
from api.station_state import ReplayStateProvider, LiveStateProvider   # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "config", "run_config.json")
ARTIFACTS_EVAL = os.path.join(ROOT, "artifacts", "eval_work")
SPEC_PATH = os.path.join(ARTIFACTS_EVAL, "processed", "feature_spec.json")
EVAL_REPORT = os.path.join(ROOT, "artifacts", "evaluation_report.json")
PROCESSED_TEST = os.path.join(ARTIFACTS_EVAL, "processed", "test.parquet")


# ---------------------------------------------------------------------------
# 靜態站點表：由 processed test.parquet 建出 {station_key → static_info}
# 供 LiveStateProvider 填補即時來源可能缺少的靜態欄位（座標/分群/型別）。
# ---------------------------------------------------------------------------
def _build_static_from_processed(processed_path):
    """唯讀：從 processed test.parquet 抽出各站靜態屬性（去重、取首見）。"""
    import pandas as pd
    CLUSTER_COLS = {c: f"cluster_{c}" for c in
                    ["平穩低波動型", "均衡流動型", "雙尖峰高流動型", "通勤到達型", "未知或新場站"]}
    TYPE_COLS = ["捷運站", "公車站", "學校", "公園", "醫院", "商圈", "停車場"]
    CAL_COLS = {"假日": "cal_假日", "國定假日": "cal_國定假日", "平日": "cal_平日"}

    def cluster_of(row):
        for c, col in CLUSTER_COLS.items():
            if col in row.index and float(row[col]) >= 0.5:
                return c
        return "未知或新場站"

    def types_of(row):
        return [t for t in TYPE_COLS if f"type_{t}" in row.index and float(row[f"type_{t}"]) >= 0.5]

    def cal_of(row):
        for c, col in CAL_COLS.items():
            if col in row.index and float(row[col]) >= 0.5:
                return c
        return None

    df = pd.read_parquet(processed_path)
    df = df.drop_duplicates("station_key", keep="first")
    out = {}
    for _, r in df.iterrows():
        out[str(r["station_key"])] = {
            "lon": float(r["經度"]), "lat": float(r["緯度"]),
            "total_docks": float(r["總車柱數"]),
            "cluster": cluster_of(r),
            "station_types": types_of(r),
            "calendar": cal_of(r),
        }
    return out


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------
class AppState:
    """啟動時載入一次模型與規格；管理回放快取與即時來源。"""

    def __init__(self, fetch_current_fn=None):
        """fetch_current_fn: callable() → list[dict]，即時站況來源。
        None = 即時模式尚未接入（明確回 503，不以回放值頂替）。"""
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.spec = load_spec(SPEC_PATH)
        self.thresholds = load_thresholds(EVAL_REPORT)
        self.source, self.mode = make_predictor(self.cfg, ARTIFACTS_EVAL)
        self.predictor = Predictor(self.spec, self.thresholds, self.source)
        # mme 與 endpoint 皆為「已接線上 SageMaker Endpoint」
        self.has_endpoint = (self.mode in ("endpoint", "mme"))

        # 回放
        self.replay_provider = None
        self._replay_error = None
        try:
            self.replay_provider = ReplayStateProvider(PROCESSED_TEST)
        except Exception as e:
            self._replay_error = f"{type(e).__name__}: {e}"

        # 即時來源
        self._live_source_env = os.environ.get("YOUBIKE_LIVE_SOURCE", "").strip()
        self._static_by_key = {}
        try:
            self._static_by_key = _build_static_from_processed(PROCESSED_TEST)
        except Exception:
            pass
        self.live_provider = LiveStateProvider(
            fetch_current=fetch_current_fn,
            static_by_key=self._static_by_key,
        )

        # 快取
        self._cache = {}
        self._cache_lock = threading.Lock()

    # ---- 回放 ---------------------------------------------------------------
    def replay(self, date_str, time_str):
        if self.replay_provider is None:
            raise RuntimeError(f"回放來源不可用：{self._replay_error}")
        key = ("replay", date_str, time_str, self.mode)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        states = self.replay_provider.states_at(date_str, time_str)
        for st in states:
            if st.get("_is_continuous", 1.0) < 0.5:
                st["history"][120] = {"ratio_bike": None, "ratio_dock": None}
        t0 = time.time()
        results = self.predictor.predict_batch(states)
        elapsed = round(time.time() - t0, 3)
        payload = {
            "mode": "replay",
            "traceable": True,
            "dataTime": f"{date_str} {time_str}",
            "dataFreshness": "歷史回放（後端以本次模型工件重新推論該時點站況）",
            "source": "processed test.parquet（6 月，可追溯本次四個 -r2 模型工件）",
            "modelVersion": self.source.version(),
            "warningThresholds": self.thresholds,
            "stationCount": len(results),
            "computeSeconds": elapsed,
            "stations": results,
        }
        with self._cache_lock:
            self._cache[key] = payload
        return payload

    # ---- 即時 ----------------------------------------------------------------
    def live_readiness(self):
        """詳細就緒狀態（不含 stations）。missing 列出所有阻擋項目。"""
        missing = []
        if not self.has_endpoint:
            missing.append("尚無 InService 的 SageMaker Endpoint（或未設定環境變數 YOUBIKE_SM_ENDPOINTS）")
        if not self.live_provider.available():
            missing.append("尚未接入持續更新的即時站況來源（fetch_current 未注入）")
        prov_status = self.live_provider.status()
        buffered = prov_status.get("bufferedStations", 0)
        if self.live_provider.available() and buffered == 0:
            missing.append("即時站況來源已接入但尚無任何觀測緩衝（需先 poll_once() 至少一次）")
        return {
            "available": not missing,
            "hasEndpoint": self.has_endpoint,
            "liveSourceConfigured": self.live_provider.available(),
            "bufferedStations": buffered,
            "missing": missing,
            "providerStatus": prov_status,
        }

    def live(self):
        """即時推論主流程。就緒才跑；否則拋錯。"""
        r = self.live_readiness()
        if not r["available"]:
            raise LiveNotReadyError(r["missing"], r)
        # 由 live_provider 組 StationState 清單
        states = self.live_provider.states()
        if not states:
            raise LiveNotReadyError(["快照緩衝有站點資料但 states() 回傳空"], r)
        t0 = time.time()
        results = self.predictor.predict_batch(states)
        elapsed = round(time.time() - t0, 3)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "mode": "live",
            "available": True,
            "dataTime": now_str,
            "dataFreshness": "即時推論（後端呼叫 SageMaker Endpoint，並以 30/60/90/120 分快照建立 lag 特徵）",
            "source": "即時站況 + SnapshotBuffer",
            "modelVersion": self.source.version(),
            "warningThresholds": self.thresholds,
            "stationCount": len(results),
            "computeSeconds": elapsed,
            "stations": results,
            "liveStatus": r,
        }


class LiveNotReadyError(Exception):
    def __init__(self, missing, status):
        super().__init__(str(missing))
        self.missing = missing
        self.status = status


APP = None


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "YouBikeAPI/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)
        try:
            if path == "/api/health":
                self._send(200, {
                    "status": "ok", "mode": APP.mode,
                    "hasEndpoint": APP.has_endpoint,
                    "replayAvailable": APP.replay_provider is not None,
                    "liveReadiness": APP.live_readiness(),
                    "runId": APP.cfg["run_id"],
                })
            elif path == "/api/meta":
                m = APP.predictor.meta()
                m["liveReadiness"] = APP.live_readiness()
                if APP.replay_provider is not None:
                    m["replayDates"] = APP.replay_provider.available_dates()
                self._send(200, m)
            elif path == "/api/replay":
                date_str = qs.get("date", ["2026-06-29"])[0]
                time_str = qs.get("time", ["06:30"])[0]
                self._send(200, APP.replay(date_str, time_str))
            elif path == "/api/live":
                try:
                    self._send(200, APP.live())
                except LiveNotReadyError as e:
                    # 明確 503 + 缺什麼，不以回放值冒充即時
                    self._send(503, {
                        "mode": "live", "available": False,
                        "reason": "即時模式尚未就緒",
                        "missing": e.missing,
                        "liveStatus": e.status,
                        "hint": "回放模式可用：GET /api/replay",
                    })
            elif path == "/api/live/status":
                r = APP.live_readiness()
                self._send(200 if r["available"] else 503, r)
            else:
                self._send(404, {"error": "not found", "path": path})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


def build_app(fetch_current_fn=None):
    global APP
    APP = AppState(fetch_current_fn=fetch_current_fn)
    return APP


def _maybe_make_live_fetch():
    """今日即時來源（NTPC ?size=3000）注入。預設關閉。

    只有設定環境變數 YOUBIKE_ENABLE_LIVE=1 時才啟用，且：
      - 只在記憶體累積快照（SnapshotBuffer），不寫任何磁碟或雲端。
      - 冷啟動需累積過去 30/60/90/120 分鐘（約 2 小時）快照，未齊前 /api/live 各站顯示資料不足。
    若要『持久保存快照跨重啟』需本地寫檔——此涉及本地持久寫入，依指示先不實作、等使用者決定。
    """
    if os.environ.get("YOUBIKE_ENABLE_LIVE", "").strip() != "1":
        return None
    try:
        import pandas as pd
        from api.ntpc_live import build_station_registry, make_fetch_current
        train_keys = set(pd.read_parquet(PROCESSED_TEST, columns=["station_key"])["station_key"].unique())
        registry = build_station_registry(train_keys)
        return make_fetch_current(registry)
    except Exception as e:
        print(f"[live] 無法建立即時來源，維持 /api/live 503：{type(e).__name__}: {e}", flush=True)
        return None


def main():
    fetch_fn = _maybe_make_live_fetch()
    build_app(fetch_current_fn=fetch_fn)
    port = int(os.environ.get("YOUBIKE_API_PORT", "8000"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    live_note = "on(in-memory,需2小時快照)" if fetch_fn else "off(預設,/api/live=503)"
    print(f"YOUBIKE_API_LISTENING port={port} mode={APP.mode} hasEndpoint={APP.has_endpoint} live={live_note}",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
