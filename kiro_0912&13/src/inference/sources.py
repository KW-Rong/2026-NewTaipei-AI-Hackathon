# -*- coding: utf-8 -*-
"""推論來源抽象層：四個模型『比例輸出』從哪裡來。

兩種 backend，皆回傳同一介面 predict_ratios(vectors_by_target) -> {target: [ratio,...]}：

  1. LocalArtifactPredictor
     - 載入本次 RUN 四個模型工件的 model.json（本地 artifacts/eval_work/{target}/），
       用 xgboost 直接推論。用於：離線開發、單元測試、以及『尚未部署 Endpoint 時』的
       後端推論來源（結果仍完全可追溯本次四個 -r2 模型工件）。
     - 注意：這是用『模型工件』在本機算，不是呼叫 SageMaker 線上 Endpoint。

  2. SageMakerEndpointClient
     - 呼叫真實的 SageMaker Runtime Endpoint（invoke_endpoint）。
     - 僅在環境變數提供 Endpoint 名稱且帳號有 InService Endpoint 時才可用。
     - 憑證來自伺服器端環境（boto3 預設鏈：環境變數/IAM Role/設定檔），
       絕不寫入程式碼、前端或 log。

工廠 make_predictor() 依環境變數決定用哪個 backend；預設 local。
"""
import json
import os
import tarfile

TARGETS = ["bike-30", "dock-30", "bike-60", "dock-60"]


class LocalArtifactPredictor:
    """用本地四個模型工件 model.json 推論（可追溯本次 -r2 模型）。"""

    def __init__(self, artifacts_dir, run_id):
        import xgboost as xgb  # 延遲載入
        self._xgb = xgb
        self.run_id = run_id
        self.kind = "local_artifact"
        self.boosters = {}
        self.best_iter = {}
        self.model_source = {}
        for t in TARGETS:
            mj = os.path.join(artifacts_dir, t, "model.json")
            if not os.path.exists(mj):
                # 嘗試從 model.tar.gz 解出
                tarp = os.path.join(artifacts_dir, t, "model.tar.gz")
                if os.path.exists(tarp):
                    with tarfile.open(tarp, "r:gz") as tf:
                        tf.extractall(os.path.join(artifacts_dir, t))
            if not os.path.exists(mj):
                raise FileNotFoundError(f"找不到 {t} 的 model.json：{mj}")
            b = xgb.Booster()
            b.load_model(mj)
            meta_p = os.path.join(artifacts_dir, t, "train_meta.json")
            bi = None
            if os.path.exists(meta_p):
                with open(meta_p, "r", encoding="utf-8") as f:
                    bi = json.load(f).get("best_iteration")
            self.boosters[t] = b
            self.best_iter[t] = bi
            self.model_source[t] = mj

    def version(self):
        return {
            "kind": self.kind,
            "runId": self.run_id,
            "jobs": {t: f"{self.run_id}-{t}-r2" for t in TARGETS},
            "note": "以本次四個 -r2 模型工件在後端本機推論（非線上 SageMaker Endpoint）。",
        }

    def predict_ratios(self, vectors_by_target):
        import numpy as np
        out = {}
        for t in TARGETS:
            vecs = vectors_by_target.get(t) or []
            if not vecs:
                out[t] = []
                continue
            # 模型以帶 feature_names 的 DMatrix 訓練；推論須帶相同 feature_names。
            fnames = self.boosters[t].feature_names
            dm = self._xgb.DMatrix(np.asarray(vecs, dtype=np.float32), feature_names=fnames)
            bi = self.best_iter.get(t)
            if bi is not None:
                pred = self.boosters[t].predict(dm, iteration_range=(0, int(bi) + 1))
            else:
                pred = self.boosters[t].predict(dm)
            out[t] = [float(x) for x in pred]
        return out


class SageMakerEndpointClient:
    """呼叫真實 SageMaker Runtime Endpoint。目前帳號尚無 InService Endpoint，故預設不啟用。

    需要環境變數：
      YOUBIKE_SM_ENDPOINTS='{"bike-30":"ep-...","dock-30":"...","bike-60":"...","dock-60":"..."}'
      （或單一多模型 Endpoint 的對應設定）
    憑證由 boto3 預設鏈提供（伺服器端），不寫入程式。
    """

    def __init__(self, endpoint_map, region, run_id, profile=None):
        import boto3
        from botocore.config import Config
        sess = boto3.Session(profile_name=profile, region_name=region) if profile \
            else boto3.Session(region_name=region)
        self._rt = sess.client("sagemaker-runtime",
                               config=Config(connect_timeout=5, read_timeout=30,
                                             retries={"max_attempts": 2}))
        self.endpoint_map = endpoint_map
        self.run_id = run_id
        self.kind = "sagemaker_endpoint"

    def version(self):
        return {
            "kind": self.kind,
            "runId": self.run_id,
            "endpoints": self.endpoint_map,
            "note": "呼叫線上 SageMaker Endpoint。",
        }

    def predict_ratios(self, vectors_by_target):
        import csv
        import io
        out = {}
        for t in TARGETS:
            vecs = vectors_by_target.get(t) or []
            ep = self.endpoint_map.get(t)
            if not vecs:
                out[t] = []
                continue
            if not ep:
                raise RuntimeError(f"未設定 {t} 的 Endpoint 名稱")
            buf = io.StringIO()
            csv.writer(buf).writerows(vecs)
            resp = self._rt.invoke_endpoint(EndpointName=ep, ContentType="text/csv",
                                            Body=buf.getvalue().encode("utf-8"))
            body = resp["Body"].read().decode("utf-8").strip()
            out[t] = [float(x) for x in body.replace("\n", ",").split(",") if x.strip()]
        return out


class SageMakerMMEClient:
    """呼叫單一 Multi-Model Endpoint（MME），以 TargetModel 區分四個模型。

    需要環境變數：
      YOUBIKE_SM_MME_ENDPOINT='youbike-20260912-122927-mme'
    TargetModel 對應（預設 {target}.tar.gz）可用 YOUBIKE_SM_MME_TARGETS 覆寫（JSON）。
    憑證由 boto3 預設鏈提供（伺服器端），不寫入程式。
    """

    def __init__(self, endpoint_name, region, run_id, target_models=None, profile=None):
        import boto3
        from botocore.config import Config
        sess = boto3.Session(profile_name=profile, region_name=region) if profile \
            else boto3.Session(region_name=region)
        self._rt = sess.client("sagemaker-runtime",
                               config=Config(connect_timeout=5, read_timeout=60,
                                             retries={"max_attempts": 2}))
        self.endpoint_name = endpoint_name
        self.run_id = run_id
        self.target_models = target_models or {t: f"{t}.tar.gz" for t in TARGETS}
        self.kind = "sagemaker_mme"

    def version(self):
        return {
            "kind": self.kind,
            "runId": self.run_id,
            "endpoint": self.endpoint_name,
            "targetModels": self.target_models,
            "note": "呼叫線上 SageMaker 多模型 Endpoint（MME，以 TargetModel 區分四模型）。",
        }

    def predict_ratios(self, vectors_by_target):
        import csv
        import io
        out = {}
        for t in TARGETS:
            vecs = vectors_by_target.get(t) or []
            if not vecs:
                out[t] = []
                continue
            tm = self.target_models.get(t)
            if not tm:
                raise RuntimeError(f"未設定 {t} 的 TargetModel")
            buf = io.StringIO()
            csv.writer(buf).writerows(vecs)   # 多列一次送（分批），無表頭
            resp = self._rt.invoke_endpoint(
                EndpointName=self.endpoint_name, TargetModel=tm,
                ContentType="text/csv", Body=buf.getvalue().encode("utf-8"))
            body = resp["Body"].read().decode("utf-8").strip()
            # 逐列解析：回應可能是每列一個 "[0.35]" / 純數字 / 逗號分隔。
            # 嚴格要求「輸出列數 == 輸入列數」，否則報錯拒絕整批（不靜默截斷/錯位回填）。
            import re
            expected = len(vecs)
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            vals = None
            if len(lines) == expected:
                # 每列一個預測：逐列抽第一個浮點數（保留原始比例，不做任何正規化）
                parsed = []
                for ln in lines:
                    m = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", ln)
                    if not m:
                        raise RuntimeError(
                            f"MME {tm} 第 {len(parsed)+1} 列回應無法解析為數值：{ln[:80]!r}")
                    parsed.append(float(m.group(0)))
                vals = parsed
            else:
                # 回應非逐列格式（例如整包一行）：全量抽數字後嚴格比對數量
                nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", body)
                if len(nums) != expected:
                    raise RuntimeError(
                        f"MME {tm} 回應數量不符：送入 {expected} 列，取得 {len(nums)} 個數值。"
                        f"拒絕此批結果以避免錯位回填。回應前 120 字：{body[:120]!r}")
                vals = [float(x) for x in nums]
            if len(vals) != expected:
                raise RuntimeError(
                    f"MME {tm} 解析後數量 {len(vals)} != 送入 {expected}，拒絕此批。")
            out[t] = vals   # 原始模型比例，未做加總正規化
        return out


def make_predictor(cfg, artifacts_dir):
    """依環境變數選 backend。優先序：MME > 多 Endpoint > 本機工件。

    - YOUBIKE_SM_MME_ENDPOINT 有設定 → SageMakerMMEClient（單一 MME，以 TargetModel 區分）。
    - YOUBIKE_SM_ENDPOINTS 有設定    → SageMakerEndpointClient（每模型一個 Endpoint）。
    - 皆無                           → LocalArtifactPredictor（本機工件，可追溯，非線上）。
    回傳 (predictor, mode_str)。mode_str: 'mme' | 'endpoint' | 'local_artifact'
    """
    run_id = cfg["run_id"]
    profile = os.environ.get("YOUBIKE_AWS_PROFILE")
    mme_env = os.environ.get("YOUBIKE_SM_MME_ENDPOINT", "").strip()
    if mme_env:
        tm_env = os.environ.get("YOUBIKE_SM_MME_TARGETS", "").strip()
        target_models = None
        if tm_env:
            try:
                target_models = json.loads(tm_env)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"YOUBIKE_SM_MME_TARGETS 不是合法 JSON：{e}")
        return SageMakerMMEClient(mme_env, cfg["aws"]["region"], run_id,
                                  target_models=target_models, profile=profile), "mme"
    ep_env = os.environ.get("YOUBIKE_SM_ENDPOINTS", "").strip()
    if ep_env:
        try:
            ep_map = json.loads(ep_env)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"YOUBIKE_SM_ENDPOINTS 不是合法 JSON：{e}")
        return SageMakerEndpointClient(ep_map, cfg["aws"]["region"], run_id,
                                       profile=profile), "endpoint"
    return LocalArtifactPredictor(artifacts_dir, run_id), "local_artifact"
