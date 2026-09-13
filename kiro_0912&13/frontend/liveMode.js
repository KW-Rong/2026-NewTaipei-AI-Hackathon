/* 主畫面資料來源：一律使用『自家後端 API 的真實 SageMaker MME 推論』（/api/replay）。
 *
 * 原則：
 *  - 瀏覽器只向自家後端 API 取值（預設 http://127.0.0.1:8000，可由 window.YOUBIKE_API_BASE 覆寫為同源 ''）；
 *    不放任何 AWS 金鑰、也不直接呼叫 SageMaker。
 *  - 主畫面所有預測（地圖/列表/詳情/警示/調度）都來自 MME；appData.js 僅供載入前的版型骨架，
 *    後端不可用或 API 失敗時「清空畫面並說明」，絕不以 appData.js 的靜態舊預測冒充。
 *  - 時間軸（overviewTimelineSlots[currentTimelineIndex]）切換時，按需向 /api/replay?time=<該時段>
 *    請求並快取；請求失敗清空該時段畫面、不留上一時段。
 *  - 頁首以單行顯示真實資料日期與模型來源。
 */
(function () {
    // 預設本機開發：前端 :8080、後端 :8000（不同埠），故預設指向 http://127.0.0.1:8000。
    // 公開 HTTPS 同源部署時，用 window.YOUBIKE_API_BASE 覆寫為 ''（同源 /api）。
    const API_BASE = (typeof window.YOUBIKE_API_BASE === 'string' && window.YOUBIKE_API_BASE)
        ? window.YOUBIKE_API_BASE.replace(/\/$/, '')
        : 'http://127.0.0.1:8000';

    let backendMeta = null;   // /api/meta 結果（模型版本、門檻等）
    let backendOnline = false;

    const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    const _nf = new Intl.NumberFormat('zh-TW');
    const countFormat0 = n => (n == null ? '—' : _nf.format(n));

    async function fetchJson(path, timeoutMs = 30000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const resp = await fetch(API_BASE + path, { signal: controller.signal });
            const body = await resp.json().catch(() => ({}));
            return { ok: resp.ok, status: resp.status, body };
        } finally {
            clearTimeout(timer);
        }
    }

    const MME_REPLAY_DATE = '2026-06-29';
    // 供「站點供需型態分析」頁的整日統計沿用同一回放日期（唯讀，不改變本檔行為）。
    window.YOUBIKE_REPLAY_DATE = MME_REPLAY_DATE;
    // MME 時間軸：跟隨 index.html 的 overviewTimelineSlots[currentTimelineIndex]。
    // 每個時段按需向後端請求並快取；請求失敗清空畫面、不留上一時段的預測。
    const _mmeSlotCache = new Map();   // 'HH:MM' -> { stations, body }
    let _mmeLoading = false;           // 防重入
    let _mmeLastSlot = null;

    // 目前時間軸選定的基準時段（HH:MM）。若全域尚未就緒則回退 06:30（index.html 預設）。
    function currentTimelineSlot() {
        try {
            if (typeof overviewTimelineSlots !== 'undefined'
                && typeof currentTimelineIndex === 'number'
                && overviewTimelineSlots[currentTimelineIndex]) {
                return overviewTimelineSlots[currentTimelineIndex];
            }
        } catch (e) { /* fallthrough */ }
        return '06:30';
    }

    // 就地替換全域 stationData 的內容（因為它以 const 宣告，不能重新指派，只能改內容）。
    function replaceStationData(newArr) {
        if (typeof stationData === 'undefined') return;
        stationData.length = 0;
        newArr.forEach(s => stationData.push(s));
    }

    // 後端 station → 前端統一 station 模型（供地圖/列表/詳情/警示/調度共用）。
    // 預測值一律來自 MME（station.mmePred）；lat/lng/name/district/pattern 為站點固定地理屬性。
    function toFrontendStation(bs, idx) {
        const parts = String(bs.stationKey || '').split('|');
        const preds = bs.predictions || {};
        return {
            id: 'M' + String(idx + 1).padStart(4, '0'),
            key: bs.stationKey,
            name: bs.name || (parts[2] || bs.stationKey),
            district: bs.district || (parts[1] || ''),
            pattern: bs.cluster || '未分群新站',
            lat: bs.lat, lng: bs.lon,       // 後端欄位名為 lon
            total: bs.total || 0,
            geoTags: [],                    // MME 回放未帶周邊標籤
            // MME 單點預測（存取器 bikesAt/bikeRatioAt/dockRatioAt/warningAt 會讀這個）
            mmePred: {
                status: bs.dataStatus || 'insufficient',
                bikesNow: (bs.bikesNow == null ? null : bs.bikesNow),
                total: bs.total || 0,
                // 當下比例（後端未取整，供 horizon=0 與 KPI 同源，與 immediate 事件一致）
                curBikeRatio: (bs.curBikeRatio == null ? null : bs.curBikeRatio),
                curDockRatio: (bs.curDockRatio == null ? null : bs.curDockRatio),
                p: {
                    '30': preds['30'] || null,
                    '60': preds['60'] || null,
                },
                reasons: bs.reasons || [],
                // 後端 authoritative 警示（含 immediate 目前事件與 +30/+60 未來預警）。
                // 前端一律以此為準，不自行用取整後比例重算，避免邊界（比例=0.10）翻轉。
                alerts: Array.isArray(bs.alerts) ? bs.alerts : [],
            },
        };
    }

    function rerenderAll() {
        // 重新渲染五項主畫面功能（皆讀全域 stationData）。
        if (typeof window.renderMarkers === 'function') window.renderMarkers();
        if (typeof handleStationSearch === 'function') handleStationSearch(typeof patternSearchQuery !== 'undefined' ? patternSearchQuery : '');
        if (typeof window.updateOverviewRiskCounts === 'function') window.updateOverviewRiskCounts();
        // 站點詳情：若目前選定站仍在可見集合則保留，否則選可見第一站，無站則清空。
        if (typeof window.syncPanelToVisible === 'function') {
            window.syncPanelToVisible();
        } else if (typeof stationData !== 'undefined' && stationData.length && typeof selectStationForPanel === 'function') {
            selectStationForPanel(stationData[0].id);
        }
    }

    // 清空主畫面（不留任何靜態/舊時段數字）。用於後端不可用或 API 失敗。
    function clearMainScreen(panelName, panelSub) {
        replaceStationData([]);
        if (typeof window.renderMarkers === 'function') window.renderMarkers();
        if (typeof handleStationSearch === 'function') handleStationSearch('');
        if (typeof window.updateOverviewRiskCounts === 'function') window.updateOverviewRiskCounts();
        ['panel-bikes-now', 'panel-bikes-30', 'panel-bikes-60'].forEach(id => {
            const el = document.getElementById(id); if (el) el.textContent = '—';
        });
        const nameEl = document.getElementById('panel-station-name');
        if (nameEl && panelName) nameEl.textContent = panelName;
        const subEl = document.getElementById('panel-station-sub');
        if (subEl && panelSub) subEl.textContent = panelSub;
    }

    // 後端 SageMaker MME 預測：前端只呼叫 /api/replay，把整批 MME 預測灌進五項主畫面功能。
    // 資料流：站況 → 後端組 54 維 → SageMaker MME 四模型 → 後端 JSON → 前端統一狀態 → 畫面。
    // 註：此為『歷史回放輸入（2026-06-29）』經真實 MME 推論，非今日即時。
    async function applyBackendMmeMode() {
        window.YOUBIKE_CURRENT_MODE = 'mme';
        if (!backendOnline) {
            // 後端不可用：清空畫面，不以 appData.js 靜態預測冒充。
            clearMainScreen('後端 API 未連線', '主畫面預測需要後端 SageMaker MME 服務；未連線時不顯示靜態舊預測。');
            setText('header-data-period', '後端未連線（無法取得 MME 預測）');
            setText('header-model-source', '尚未連線後端 API，未顯示任何預測');
            return;
        }
        await loadMmeForSlot(currentTimelineSlot());
    }

    // 依指定時段（HH:MM）載入 MME 預測並更新畫面。命中快取則不重打 API。
    // 失敗時清空 stationData（不留上一時段），避免以舊時段預測冒充目前時段。
    async function loadMmeForSlot(slot) {
        if (window.YOUBIKE_CURRENT_MODE !== 'mme') return;
        if (_mmeLoading) return;
        _mmeLoading = true;
        _mmeLastSlot = slot;
        setText('header-data-period', `${MME_REPLAY_DATE} ${slot}（查詢中…）`);
        try {
            let entry = _mmeSlotCache.get(slot);
            if (!entry) {
                const res = await fetchJson(`/api/replay?date=${MME_REPLAY_DATE}&time=${slot}`, 180000);
                if (!res.ok || !res.body) throw new Error('replay 回應異常 ' + res.status);
                entry = { body: res.body, stations: (res.body.stations || []).map(toFrontendStation) };
                _mmeSlotCache.set(slot, entry);
            }
            // 若使用者在等待期間又切換了時段，丟棄這次結果（只套用最新選定時段）。
            if (slot !== currentTimelineSlot()) { _mmeLoading = false; return; }

            const b = entry.body;
            const mv = b.modelVersion || {};
            const feStations = entry.stations;
            replaceStationData(feStations);

            // 站數/分群/行政區依當前 stationData 動態算（此時段可能 1562 或 1576）；
            // 頁首單行來源/時間用本次 API 回應。
            const mmeOverride = {
                dataPeriod: `${b.dataTime || (MME_REPLAY_DATE + ' ' + slot)} 歷史回放輸入（非今日即時）`,
                dataTime: (b.dataTime || '—'),
                timelineDateText: (b.dataTime || '—') + '｜SageMaker MME 推論',
                modeText: 'SageMaker MME（歷史 ' + MME_REPLAY_DATE + '）',
                modelVersionText: (mv.runId || '—') + '（' + (mv.kind || '') + '）',
                observationText: countFormat0(b.stationCount) + ' 站真實推論',
                headerSource: 'SageMaker 多模型 Endpoint 真實推論'
                    + (mv.endpoint ? `（${mv.endpoint}）` : ''),
                sourceNote: '畫面所有預測來自後端呼叫既有 SageMaker 多模型 Endpoint 的真實推論，非 appData.js 靜態值。',
            };
            if (typeof window.populateFilters === 'function') window.populateFilters();
            if (typeof window.hydrateMetadata === 'function') window.hydrateMetadata(mmeOverride);
            rerenderAll();
        } catch (e) {
            // 請求失敗：清空畫面，不保留上一時段的預測、也不退回 appData 靜態值。
            _mmeSlotCache.delete(slot);
            clearMainScreen(`${MME_REPLAY_DATE} ${slot}：查詢失敗`,
                `向後端取得該時段 MME 推論失敗（${e.message}）。已清空畫面，不以上一時段或靜態值冒充。`);
            setText('header-data-period', `${MME_REPLAY_DATE} ${slot}（查詢失敗）`);
            setText('header-model-source', 'API 失敗，未顯示任何預測（不使用靜態舊預測）');
        } finally {
            _mmeLoading = false;
        }
    }

    // 供 index.html 時間軸切換時呼叫（MME 模式才動作）。
    window.onTimelineSlotChanged = function () {
        if (window.YOUBIKE_CURRENT_MODE !== 'mme') return;
        const slot = currentTimelineSlot();
        if (slot === _mmeLastSlot && stationData.length) return;  // 同時段且已有資料，免重載
        loadMmeForSlot(slot);
    };

    // 初始化：探測後端；在線→載入真實 MME 預測；不在線→清空畫面並說明（不顯示 appData 靜態預測）。
    async function init() {
        window.YOUBIKE_CURRENT_MODE = 'mme';
        setText('header-data-period', '載入中…');
        setText('header-model-source', '正在連線後端 API…');
        try {
            const health = await fetchJson('/api/health', 4000);
            backendOnline = health.ok && health.body && health.body.status === 'ok';
            if (backendOnline) {
                const meta = await fetchJson('/api/meta', 8000);
                if (meta.ok) backendMeta = meta.body;
            }
        } catch (e) {
            backendOnline = false;
        }
        await applyBackendMmeMode();
    }

    if (document.readyState === 'complete') init();
    else window.addEventListener('load', init);
}());
