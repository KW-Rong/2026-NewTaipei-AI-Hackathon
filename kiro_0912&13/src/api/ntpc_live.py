# -*- coding: utf-8 -*-
"""新北市 YouBike 2.0 即時開放資料 adapter（已驗證 ?size=N 分頁）。

資料集：010e5b15-3823-4b20-b401-b1cf000550c5
單一請求 ?size=3000 取全量（~1606 站）。欄位：
  sno, sna(含 YouBike2.0_ 前綴), sarea, tot_quantity(容量),
  sbi_quantity(可借), bemp(可還), act(1營運/0停用), mday(YYYYMMDDTHHMMSS),
  lat, lng, scity。

fetch_current() 回傳 predictor 需要的 StationState 觀測（不含 history；
history 由 SnapshotBuffer 累積）。站號/站名對應訓練 station_key（新北市|區|站名）。

不含任何憑證。純 HTTP。
"""
import json
import re
import time
import urllib.request
from datetime import datetime

DATASET_ID = "010e5b15-3823-4b20-b401-b1cf000550c5"
DEFAULT_URL = f"https://data.ntpc.gov.tw/api/datasets/{DATASET_ID}/json?size=3000"


def _num(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def _parse_mday(s):
    """YYYYMMDDTHHMMSS → datetime。"""
    s = str(s).strip()
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})$", s)
    if m:
        g = m.groups()
        return datetime(int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))
    return None


def clean_name(sna):
    return re.sub(r"^YouBike2\.0_\s*", "", str(sna)).strip()


def build_station_registry(train_station_keys):
    """由訓練 station_key 集合建 (行政區, 站名) → station_key 索引，供對應驗證。"""
    idx = {}
    for k in train_station_keys:
        p = str(k).split("|")
        if len(p) == 3:
            idx[(p[1].strip(), p[2].strip())] = k
    return idx


def fetch_ntpc_raw(url=DEFAULT_URL, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "kiro0912/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_fetch_current(train_index, url=DEFAULT_URL):
    """回傳一個 fetch_current() callable。train_index: (區,站名)->station_key。

    只回傳能對應到訓練站點的觀測（station_key 用訓練鍵）；對應不到的站標記後仍回傳，
    但 station_key 用即時鍵並記 mapping_verified=False（predictor 端會因分群/歷史處理）。
    比例：ratio_bike = 可借/容量、ratio_dock = 可還/容量、ratio_unavailable = 1-(可借+可還)/容量。
    """
    def fetch_current():
        rows = fetch_ntpc_raw(url)
        now = datetime.now()
        out = []
        for r in rows:
            area = str(r.get("sarea", "")).strip()
            name = clean_name(r.get("sna", ""))
            total = _num(r.get("tot_quantity"))
            sbi = _num(r.get("sbi_quantity"))
            bemp = _num(r.get("bemp"))
            act = str(r.get("act", "1")).strip()
            ts = _parse_mday(r.get("mday")) or now
            skey = train_index.get((area, name))     # 對應到訓練鍵
            mapped = skey is not None
            if not skey:
                skey = f"新北市|{area}|{name}"          # 未對應：用即時鍵（新站/改名）
            ratio_bike = (sbi / total) if (total and sbi is not None and total > 0) else None
            ratio_dock = (bemp / total) if (total and bemp is not None and total > 0) else None
            ratio_unavail = None
            if total and sbi is not None and bemp is not None and total > 0:
                ratio_unavail = max(0.0, 1.0 - (sbi + bemp) / total)
            out.append({
                "station_key": skey,
                "station_id": str(r.get("sno")),
                "mapping_verified": mapped,
                "ts": ts,
                "ratio_bike": ratio_bike,
                "ratio_dock": ratio_dock,
                "ratio_unavailable": ratio_unavail if ratio_unavail is not None else 0.0,
                "total_docks": total,
                "operational": (act == "1"),
                "lon": _num(r.get("lng")),
                "lat": _num(r.get("lat")),
                # cluster/station_types/calendar 由後端靜態表補（static_by_key）
            })
        return out
    return fetch_current
