# SageMaker Endpoint 部署方案與估價（待你確認機型與運行時間後才建立）

RUN_ID：`youbike-20260912-122927`／profile `hackathon-team3`／region `us-west-2`
價格來源：AWS Pricing API 實查（`_price_check.py` → `_price_check.json`，us-west-2，SageMaker 即時推論 Hosting，On-Demand）。
**本文件僅為估價與方案；在你明確回覆採用哪個方案與運行時間前，不會建立任何會計費的資源。**

## 一、要承載的模型（已核對）
四個 `-r2` 模型工件（S3，已確認存在）＋ `feature_spec.json`（54 維／模型）：

| 模型 | S3 model.tar.gz 大小 | 門檻（5 月校正）|
|---|---|---|
| bike-30 | 1.1 MB | 0.115 |
| dock-30 | 8.8 MB | 0.105 |
| bike-60 | 1.0 MB | 0.135 |
| dock-60 | 9.2 MB | 0.105 |

四個模型都是 XGBoost（framework 1.7-1），體積小、CPU 即可、記憶體需求低。**不需要 GPU，也不需要四台機器。**

## 二、方案比較（重點：不建四台常駐機器）

### 方案 A（建議）：單一 Multi-Model Endpoint（MME），1 台實例承載四個模型
- 一個 Endpoint、一台實例，動態載入四個模型；四個模型合計 < 20MB，單台輕鬆容納。
- 呼叫時以 `TargetModel` 指定 `bike-30`/`dock-30`/`bike-60`/`dock-60`。
- **每小時只算 1 台實例費用。**

| 實例 | $/hr | 每日(24h) | 每日(8h 上班) | 備註 |
|---|---|---|---|---|
| ml.t2.medium | 0.056 | 1.34 | 0.45 | 最省；展示/低流量足夠 |
| ml.c5.large | 0.102 | 2.45 | 0.82 | 運算型，延遲更穩 |
| ml.m5.large | 0.115 | 2.76 | 0.92 | 通用型 |

### 方案 B：單一實例、多容器（Multi-Container Endpoint）或 4 個 Inference Component 於 1 台
- 同樣**1 台實例**承載四個模型，各自獨立容器/元件；隔離度較高、稍複雜。
- 費用與方案 A 同級（按 1 台實例計）。

### 方案 C（不建議、你也已排除）：四個獨立 Endpoint = 四台常駐機器
- 例如 4 × ml.m5.large = 4 × 0.115 = **0.46/hr、24h/日約 11.04**。四倍花費，無必要。

> 建議採 **方案 A + ml.t2.medium 或 ml.c5.large**。以 ml.t2.medium、每日 8 小時計，一天約 **US$0.45**；24 小時常開約 **US$1.34/日**。

## 三、停止計費方式
即時推論 Endpoint **只要存在就按時計費**（不論有無流量）。停止計費＝**刪除 Endpoint**：
1. `aws sagemaker delete-endpoint --endpoint-name <name>`（停止實例計費，最主要）
2. `aws sagemaker delete-endpoint-config --endpoint-config-name <name>`
3. （選）`aws sagemaker delete-model --model-name <name>`（Model 物件本身不計費）
- S3 上的 model.tar.gz 留著只有極少 S3 儲存費（MB 級，可忽略）。
- 建議：展示/測試用時再開，測完即刪；或用排程於下班時間自動刪除。
- 提醒：**Serverless Inference** 可做到「無流量不計費」，但 XGBoost 內建容器不一定支援 MME+Serverless 組合；若要零閒置成本，可改「單模型 Serverless × 4」評估，我可另出估價。

## 四、所需最小權限（供你或管理員確認 participant role 是否具備）
建立/測試 Endpoint 最小動作（資源可限縮到本 RUN 的命名前綴與 bucket）：
- `sagemaker:CreateModel`、`sagemaker:CreateEndpointConfig`、`sagemaker:CreateEndpoint`
- `sagemaker:DescribeEndpoint`、`sagemaker:DescribeEndpointConfig`、`sagemaker:DescribeModel`
- `sagemaker:DeleteEndpoint`、`sagemaker:DeleteEndpointConfig`、`sagemaker:DeleteModel`（停止計費用）
- `sagemaker-runtime:InvokeEndpoint`（推論測試用）
- `iam:PassRole`（把既有的 `youbike-sagemaker-exec-role` 傳給 Endpoint；**我不會建立或修改 IAM role**，只使用既有 role）
- 讀取模型工件：`s3:GetObject` on `s3://sagemaker-ntpc-youbike-.../runs/youbike-20260912-122927/models/*`

> 若建立時遇到 `AccessDenied`（例如 participant role 無 `CreateEndpoint` 或 `iam:PassRole`），我會**停止並回報需人工開啟哪一項**，不自行擴權、不改 IAM。

## 五、我需要你確認的項目（回覆後才動手）
1. 採用**方案 A（MME，1 台）**還是其他？
2. 機型：**ml.t2.medium（最省）** / ml.c5.large / ml.m5.large？
3. 運行時間：測完即刪／每日 8 小時／24 小時常開？
4. 是否同意使用既有 `youbike-sagemaker-exec-role`（`iam:PassRole`）？

你確認後，我才會：建立 Model → EndpointConfig → Endpoint，輪詢至 `InService`，用相同特徵列比較 Endpoint 與本機四模型輸出，留下 `InvokeEndpoint` 成功紀錄；測完依你指示刪除以停止計費。

## 六、目前狀態
- 尚無 InService Endpoint（重查 `endpoints=[]`）。
- 後端已能以「本地模型工件」推論（可追溯，非線上）；`SageMakerEndpointClient` 已備妥，設定 `YOUBIKE_SM_ENDPOINTS` 後即可切換為線上 Endpoint。
