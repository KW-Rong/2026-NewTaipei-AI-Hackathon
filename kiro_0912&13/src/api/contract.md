# YouBike 後端推論 API 契約

本次 RUN：`youbike-20260912-122927`。四個模型：`bike-30`、`dock-30`、`bike-60`、`dock-60`（XGBoost 比例回歸，輸出可借／可還**比例**，非機率）。

設計原則
- 前端**只呼叫本 API**，不在瀏覽器放 AWS 憑證，也不直接呼叫 SageMaker。
- 後端**批次**推論整批站點並**快取**，不由每位訪客逐站呼叫模型。
- **歷史回放**與**即時預測**分開；回放結果可追溯本次四個 `-r2` 模型工件，不以 `appData.js` 回放值冒充即時。
- 站況歷史不足（過去 30／60／90／120 分任一缺、或 2 小時不連續）→ 該站 `dataStatus="insufficient"`，**不輸出預測**。
- 比例→車數：`round(比例 × 總車柱數)`；`bikes + docks` 不超過容量（配對正規化）。
- 預警：預測可借比例 ≤ `bike` 門檻→缺車；預測可還比例 ≤ `dock` 門檻→滿站。四向門檻為 5 月校正值。

## 推論來源（後端 backend）
| backend | 何時採用 | 說明 |
|---|---|---|
| `local_artifact` | 預設（目前無 Endpoint） | 用本地四個模型工件 `model.json` 在**後端**推論；仍可追溯本次 `-r2` 模型，但**非線上 SageMaker Endpoint** |
| `sagemaker_endpoint` | 設 `YOUBIKE_SM_ENDPOINTS` 且帳號有 InService Endpoint | 呼叫線上 Endpoint（`invoke_endpoint`）；憑證由伺服器端環境提供 |

## 端點

### GET /api/health
```json
{"status":"ok","mode":"local_artifact","hasEndpoint":false,"replayAvailable":true,"runId":"youbike-20260912-122927"}
```

### GET /api/meta
回模型版本、四向門檻、事件定義、分群、特徵順序、`liveStatus`、可用回放日期。
```json
{
  "modelVersion": {"kind":"local_artifact","runId":"...","jobs":{"bike-30":"...-bike-30-r2",...}},
  "warningThresholds": {"bike-30":0.115,"dock-30":0.105,"bike-60":0.135,"dock-60":0.105},
  "eventDefinition": "當下比例<=0.10為即時事件；未來預警為預測比例<=各向門檻",
  "clusters": ["平穩低波動型","均衡流動型","雙尖峰高流動型","通勤到達型","未知或新場站"],
  "horizons": [30,60],
  "liveStatus": {"liveAvailable":false,"hasEndpoint":false,"missing":[...]},
  "replayDates": ["2026-06-01", ..., "2026-06-30"]
}
```

### GET /api/replay?date=YYYY-MM-DD&time=HH:MM
歷史回放：後端**重新推論**該時點整批站況（資料流真、模型真，站況時間為歷史）。
```json
{
  "mode": "replay",
  "traceable": true,
  "dataTime": "2026-06-29 06:30",
  "dataFreshness": "歷史回放（後端以本次模型工件重新推論該時點站況）",
  "source": "processed test.parquet（6 月，可追溯本次四個 -r2 模型工件）",
  "modelVersion": {...},
  "warningThresholds": {...},
  "stationCount": 1576,
  "computeSeconds": 0.42,
  "stations": [ /* StationResult */ ]
}
```

`StationResult`
```json
{
  "stationKey": "新北市|三峽區|三峽中山公園",
  "cluster": "平穩低波動型",
  "total": 15,
  "curTime": "2026-06-29 06:30:00",
  "dataAgeSec": 0,
  "dataStatus": "ok",            // ok | insufficient | stale
  "operational": true,
  "bikesNow": 6,                  // 當下觀測車數；資料不足時可為 null
  "predictions": {
    "30": {"bikeRatio":0.407,"dockRatio":0.531,"bikes":6,"docks":8},
    "60": {"bikeRatio":0.413,"dockRatio":0.531,"bikes":6,"docks":8}
  },
  "alerts": [
    {"kind":"shortage","horizon":30,"ratio":0.11,"threshold":0.115,"immediate":false}
  ],
  "reasons": []                   // dataStatus 非 ok 時，列出原因
}
```
- `dataStatus="insufficient"`：`predictions={}`、`alerts=[]`，`reasons` 說明缺哪段歷史。
- `dataStatus="stale"`：資料過期（`data_age_sec > staleAfterSec`），不產生預測。

### GET /api/live
即時預測。未接入即時站況來源或無 Endpoint → **503**（不以回放冒充）：
```json
{"mode":"live","available":false,"reason":"即時模式尚未就緒",
 "missing":["尚無 InService 的 SageMaker 推論 Endpoint...","尚未接入持續更新的即時站況來源..."],
 "hint":"回放模式可用：GET /api/replay"}
```
具備 Endpoint + 即時來源後，`/api/live` 才會回實際即時推論（介面已預留）。

## 環境變數（皆不含憑證）
- `YOUBIKE_SM_ENDPOINTS`：JSON，設定則走線上 Endpoint。
- `YOUBIKE_LIVE_SOURCE`：即時站況來源設定，未設則 `/api/live` 回 503。
- `YOUBIKE_API_PORT`：預設 8000。
- `YOUBIKE_AWS_PROFILE`：伺服器端 boto3 profile（選用；憑證不進程式）。

## StationState（即時來源需提供給後端的每站輸入）
```json
{
  "station_key":"新北市|區|站名","total_docks":15,"lon":121.36,"lat":24.93,
  "cluster":"平穩低波動型","station_types":["公園"],"calendar":"平日",
  "ratio_bike":0.40,"ratio_dock":0.53,"ratio_unavailable":0.0,
  "history":{"30":{"ratio_bike":..,"ratio_dock":..},"60":{...},"90":{...},"120":{...}},
  "cur_time":"2026-06-29 06:30:00","operational":true,"data_age_sec":30
}
```
