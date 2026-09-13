# 37 個「未分群新站」處理決策紀錄

RUN_ID：`youbike-20260912-122927`／profile `hackathon-team3`／region `us-west-2`
查核腳本：`_check_clustering.py`　證據：`_check_clustering.json`（唯讀，未讀昨天地端訓練 CSV）

## 結論
**37 站全部維持「未分群新站」，作為資料狀態，不硬分為任何 K-Means 型態。**

理由（三項皆成立）：

1. **無同實體站可繼承**
   - 37 站在本次 processed `train.parquet`（1–4 月）中，**沒有任何相同 `station_key`** 的已分群紀錄（`n_with_exact_train_cluster = 0`）。
   - 名稱相似的 4 站經人工檢視，皆為**不同實體站**，不可繼承：
     | 未分群站 | 相似訓練站 | 判定 |
     |---|---|---|
     | 捷運臺北大學站(1號出口) | 臺北大學(音律資訊大樓) | 不同地點（捷運出口 vs 校內大樓）|
     | 捷運臺北大學站(2號出口) | 臺北大學(音律資訊大樓) | 同上 |
     | 捷運頂埔站(3號出口) | 捷運頂埔站(2號出口)／(福田路) | 不同出口=不同租賃站；且相似站分屬不同群，無法確定 |
     | 石碇高中 | 石碇 | 不同站（高中 vs 地名點）|
   - 多數為 2026 年才通車的三鶯線捷運站（捷運三峽站、橫溪站、龍埔站、鶯歌車站、國華站、永吉公園站、陶瓷老街站、鶯桃福德站等），1–4 月訓練期不存在 → 確為**真正新站**。

2. **K-Means 分群工件未保存**
   - S3 `runs/youbike-20260912-122927/` 只有 14 個物件：`code/`、`eval/`、`models/`（四個 `-r2` model.tar.gz）、`processed/`（train/validation/test.parquet、feature_spec.json、quality_report.json）。
   - **沒有任何** K-Means 模型、標準化器（scaler）、分群中心（centroid）或分群特徵工件（`clustering_related_objects = []`）。
   - 分群在 processed 階段是以「1–4 月訓練站點 → 分群」的對照套用（見 `feature_engineering.py` 的 `build_maps`），對照本身**不含**可重算新站分群的 K-Means 模型與標準化器。

3. **新站行為資料不足**
   - 這些新站在 6 月才出現，缺乏足夠歷史行為，即使有 K-Means 工件也不宜貿然分群。

## 模型輸入的正確性
- 四個 XGBoost 模型的 `feature_spec` 本就有 `cluster_未知或新場站` one-hot 欄位；未分群新站以此編碼進模型，**模型仍可對其推論**（不影響推論能力），只是不宣稱其屬於四型之一。

## 前端呈現（已實作為動態計算）
- `appIntegration.js`：四群統計、未分群數、地圖點、下拉選單、badge **一律由當前 `stationData` 動態計算**（`computeClusterCounts()`），不使用 `meta.clusterCounts` 寫死值。
- 未分群新站固定排在四型之後，標示「（… 站，資料狀態）」，不併入四型、不作為第五種 K-Means 型態。
- 測試：`tests/test_frontend_wiring.py::test_cluster_counts_dynamic_not_hardcoded`、`::test_unclustered_not_fifth_cluster`。

## 若日後要為新站分群，需要的前提（需另行提供）
1. 原始 K-Means 分群模型 + 標準化器（scaler）工件，或可重現分群的完整特徵定義。
2. 新站足夠的歷史行為資料（例如至少數週穩定運行）。
3. 或由主辦方提供可驗證的「改名／改站號 → 舊站」官方對照表（才可判定同實體站繼承舊分群）。
