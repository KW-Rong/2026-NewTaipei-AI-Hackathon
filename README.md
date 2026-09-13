# 2026 New Taipei AI Hackathon
## 新北市 YouBike 智慧供需預測與調度系統

AI智慧城市黑客松競賽｜隊伍：貝殼幣富豪俱樂部

本專案透過新北市 YouBike 歷史資料、場站特徵與機器學習模型，預測未來場站供需狀況，並提供缺車／滿柱警示與調度資訊。

---

## 專案目標

YouBike 場站在不同時間與地點會出現供需不平衡，例如：

- 無車可借
- 無位可還
- 尖峰時段大量借車或還車
- 不同類型場站具有不同的使用模式

本專案透過歷史資料分析、場站分類、特徵工程與 XGBoost 模型，預測未來 30 分鐘與 60 分鐘的場站供需狀況，並將預測結果提供給前端系統進行視覺化與警示。

---

## 系統流程

原始 YouBike 資料  
↓  
站名與資料異常修正  
↓  
新增日曆與場站類型特徵  
↓  
K-Means 場站供需分群  
↓  
建立模型訓練／驗證／測試資料  
↓  
XGBoost 模型訓練  
↓  
30 / 60 分鐘供需預測  
↓  
缺車／滿柱警示  
↓  
前端視覺化呈現

---

## 模型與資料切分

本專案主要使用 XGBoost 建立 YouBike 場站供需預測模型。

資料依月份進行時間切分：

- Training Set：1–4 月
- Validation Set：5 月
- Test Set：6 月

預測時間：

- 未來 30 分鐘
- 未來 60 分鐘

預測內容包含場站未來的可借車與可還位供需狀況，並進一步轉換為系統警示資訊。

---

## 場站 K-Means 分群

為描述不同 YouBike 場站的供需特性，本專案建立場站供需特徵，並透過 K-Means 進行場站分群。

Repository 中保留：

- K 值評估
- PCA 分群分布
- 分群特徵熱圖
- 平日 24 小時供需曲線
- 場站供需分群特徵
- 場站與分群對照表
- K-Means 分析 Notebook

相關檔案位於：`kmeans分群資料/`

---

## 特徵工程

除原始 YouBike 場站資訊外，本專案加入額外特徵，包括：

- 日曆相關特徵
- 場站類型
- 場站供需分群
- 時間相關特徵
- 歷史場站供需狀態

相關資料處理程式與說明位於：`欄位新增資料/`

其中包含：

- 六個月資料欄位新增 Notebook
- YouBike 場站 300 公尺類型對照
- 日曆與場站標記說明表

---

## 系統程式

最終系統相關程式位於：`kiro_0912&13/`

主要結構：

- `config/`：系統與模型執行設定
- `docs/`：系統與部署相關文件
- `frontend/`：前端介面與資料串接
- `src/api/`：API 與即時 YouBike 資料處理
- `src/inference/`：模型推論流程
- `src/processing/`：特徵工程與資料處理
- `src/training/`：XGBoost 模型訓練
- `tests/`：API、模型推論與前端串接測試

---

## Repository 資料夾說明

### `raw data/`
原始 YouBike 歷史資料。

### `station name corrected data/`
修正跨月份場站名稱異常後的資料。

### `training data/`
前期整理完成的模型訓練資料。

### `欄位新增資料/`
日曆、場站類型等特徵新增程式與相關結果。

### `kmeans分群資料/`
K-Means 場站供需分群程式、分析結果與視覺化。

### `kiro_0912&13/`
最終模型推論、API、前端、部署及測試相關程式。

---

## 大型資料集下載

由於部分模型訓練、驗證與測試資料單檔超過 GitHub 一般 Git 的檔案大小限制，因此大型中間資料集未直接存放於 Repository。

完整大型資料包含：

- 欄位新增後 1–4 月資料
- K-Means 分群後 1–4 月資料
- XGBoost 模型訓練集（1–4 月）
- XGBoost 模型驗證集（5 月）
- XGBoost 模型測試集（6 月）

📁 **[下載完整大型資料集（Google Drive）](https://drive.google.com/drive/folders/1d2O3jNtCwcjBe3hIaTT9x-EHUl2tG5Su?usp=sharing)**

GitHub Repository 已保留完整程式碼、資料處理 Notebook、場站分群分析結果、系統程式及必要說明文件。

---

## 主要技術

- Python
- Jupyter Notebook
- Pandas
- Scikit-learn
- K-Means
- XGBoost
- AWS
- Amazon SageMaker
- HTML / JavaScript
- Git / GitHub

---

## 專案用途

本系統希望透過 YouBike 歷史供需模式與即時場站資訊，提前辨識可能發生缺車或滿柱的場站，協助進行場站監控、警示與調度決策。
