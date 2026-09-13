# -*- coding: utf-8 -*-
"""前端資料模式串接的靜態檢查（無 Node 環境下的近似驗證）。

驗證：
  1. index.html 有模式切換 UI（歷史回放 / 即時預測）並載入 liveMode.js。
  2. liveMode.js 即時模式只呼叫自家 API（fetch API_BASE + /api/...），
     不直接呼叫 SageMaker、不含 AWS 憑證/SDK。
  3. 回放與即時分開：即時失敗/未就緒時不套用 appData 回放值當即時。
  4. 前端所有檔（含 liveMode.js）無 AWS 金鑰/SDK/直接 invoke_endpoint。
  5. liveMode.js 顯示資料時間/來源/模型版本/資料不足/錯誤狀態的 DOM id 存在於 index.html。
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, "frontend")
INDEX = os.path.join(FE, "index.html")
LIVE = os.path.join(FE, "liveMode.js")
APPINT = os.path.join(FE, "appIntegration.js")
APPDATA = os.path.join(FE, "appData.js")


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


class TestFrontendWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = read(INDEX)
        cls.live = read(LIVE)
        cls.appint = read(APPINT)

    def test_mode_switch_ui_present(self):
        self.assertIn('setDataMode(\'replay\')', self.index)
        self.assertIn('setDataMode(\'live\')', self.index)
        self.assertIn('src="liveMode.js"', self.index)

    def test_live_calls_own_api_only(self):
        # 即時模式透過 fetch(API_BASE + path) 呼叫自家 API
        self.assertIn('fetch(API_BASE', self.live)
        self.assertIn('/api/live', self.live)
        self.assertIn('/api/health', self.live)

    def test_no_direct_sagemaker_or_credentials_in_frontend(self):
        forbidden = {
            "AKIA金鑰": r"AKIA[0-9A-Z]{16}",
            "accessKeyId": r"accessKeyId",
            "secretAccessKey": r"secretAccessKey",
            "aws-sdk": r"aws-sdk",
            "invoke_endpoint": r"invoke_endpoint",
            "sagemaker-runtime": r"sagemaker-runtime",
            "sagemaker.amazonaws": r"sagemaker\.[a-z0-9\-]*\.amazonaws",
        }
        for path in (INDEX, LIVE, APPINT, APPDATA):
            t = read(path)
            for label, pat in forbidden.items():
                self.assertFalse(re.search(pat, t),
                                 f"{os.path.basename(path)} 不應含 {label}")

    def test_live_does_not_fill_replay_as_live(self):
        # 即時未就緒分支必須是「尚未就緒 + 缺什麼」，不得把 appData 當即時
        self.assertIn('尚未就緒', self.live)
        self.assertIn('missing', self.live)
        # applyLiveMode 內不得引用 window.appData.stations 當即時資料
        live_body = self.live
        self.assertNotIn('appData.stations', live_body)

    def test_status_dom_ids_exist(self):
        for dom_id in ('mode-badge', 'mode-data-time', 'mode-source',
                       'mode-model-version', 'mode-status-line',
                       'mode-btn-replay', 'mode-btn-live'):
            self.assertIn(f'id="{dom_id}"', self.index, f"缺 DOM id {dom_id}")

    def test_replay_mode_still_traceable(self):
        # 回放橫幅顯示可追溯來源（沿用 appData.meta）
        self.assertIn('replayMeta', self.live)
        self.assertIn('modelVersion', self.live)

    def test_cluster_counts_dynamic_not_hardcoded(self):
        # 四群統計/未分群數由 stationData 動態計算，不用 meta.clusterCounts 寫死
        self.assertIn('computeClusterCounts', self.appint)
        # populateFilters/badge 不應再引用 meta.clusterCounts 當顯示來源
        self.assertNotIn('meta.clusterCounts', self.appint)

    def test_unclustered_not_fifth_cluster(self):
        # 未分群新站固定排在四型之後、標為資料狀態，不併入 MODEL_CLUSTERS
        self.assertIn('MODEL_CLUSTERS', self.appint)
        self.assertIn('資料狀態', self.appint)
        # MODEL_CLUSTERS 只含四型
        m = re.search(r"MODEL_CLUSTERS\s*=\s*\[([^\]]*)\]", self.appint)
        self.assertIsNotNone(m)
        self.assertNotIn('未分群新站', m.group(1))
        self.assertNotIn('未知或新場站', m.group(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
